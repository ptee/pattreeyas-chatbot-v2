"""
MCP Server for CV Database Access
Provides standardized tools for querying PostgreSQL and Qdrant vector DB

Refactored to use:
- ConfigManager (config.py) for unified configuration
- PostgreSQLManager and QdrantManager (db_manager.py) for database operations
- Custom exceptions (exceptions.py) for better error handling
"""

import json
import logging
from typing import Dict, List, Any, Optional

from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import get_config, ConfigManager
from db_manager import get_postgres_manager, get_qdrant_manager, PostgreSQLManager, QdrantManager
from exceptions import (
    PostgreSQLConnectionError,
    QdrantConnectionError,
    DatabaseQueryError,
    CVNotFoundError,
    InvalidUUIDError,
    MCPServerError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DIAGNOSTIC UTILITIES
# ============================================================================

def diagnose_cv_availability() -> Dict[str, Any]:
    """
    Diagnose CV data availability and provide actionable feedback.

    Returns:
        dict: Status and diagnosis information
    """
    diagnosis = {
        "status": "unknown",
        "cv_id": None,
        "message": "",
        "next_steps": [],
        "troubleshooting_tips": ""
    }

    try:
        config = get_config()
        cv_id = config.get_cv_id()
        diagnosis["status"] = "success"
        diagnosis["cv_id"] = cv_id
        diagnosis["message"] = f"✓ CV ID found and validated: {cv_id[:8]}..."
        diagnosis["troubleshooting_tips"] = "System is healthy. All CV data loaded successfully."
        return diagnosis

    except ValueError as e:
        error_msg = str(e)

        if "No CV data found" in error_msg:
            diagnosis["status"] = "empty_database"
            diagnosis["message"] = "❌ cv_metadata table is EMPTY (no CV records)"
            diagnosis["next_steps"] = [
                "1. Run schema creation: python create_tables.py",
                "2. Run data ingestion: python db_ingestion.py",
                "3. Verify cv.json exists in current directory",
                "4. Check OpenAI API key is set: echo $OPENAI_API_KEY"
            ]
            diagnosis["troubleshooting_tips"] = (
                "This happens when create_tables.py was run but db_ingestion.py was not. "
                "Run db_ingestion.py to populate the database with CV data."
            )

        elif "not a valid UUID" in error_msg:
            diagnosis["status"] = "invalid_uuid"
            diagnosis["message"] = "❌ Invalid UUID format in cv_metadata.id column"
            diagnosis["next_steps"] = [
                "1. Check corrupted data: psql -c \"SELECT * FROM cv_metadata;\"",
                "2. Clear bad data: python -c \"from create_tables import *; " +
                    "conn = get_connection(); conn.cursor().execute('DELETE FROM cv_metadata'); " +
                    "conn.commit(); conn.close()\"",
                "3. Re-ingest: python db_ingestion.py"
            ]
            diagnosis["troubleshooting_tips"] = (
                "UUID validation failed. This usually means data was corrupted during ingestion. "
                "Delete the bad record and re-ingest from cv.json."
            )

        elif "relation" in error_msg or "does not exist" in error_msg:
            diagnosis["status"] = "schema_missing"
            diagnosis["message"] = "❌ cv_metadata table does NOT EXIST (schema not created)"
            diagnosis["next_steps"] = [
                "1. Create schema: python create_tables.py",
                "2. Verify creation: psql -c \"SELECT table_name FROM information_schema.tables WHERE table_schema='public';\"",
                "3. Ingest data: python db_ingestion.py"
            ]
            diagnosis["troubleshooting_tips"] = (
                "Schema was never created. Run create_tables.py first to create all required tables."
            )

        elif "could not connect" in error_msg or "connection refused" in error_msg:
            diagnosis["status"] = "connection_failed"
            diagnosis["message"] = "❌ PostgreSQL connection FAILED"
            diagnosis["next_steps"] = [
                "1. Check PostgreSQL status: systemctl status postgresql",
                "2. Test connection: psql -h 212.132.98.160 -U pt -d pt_db -c \"SELECT 1;\"",
                "3. Verify credentials in .streamlit/secrets.toml",
                "4. Check firewall: telnet 212.132.98.160 5432"
            ]
            diagnosis["troubleshooting_tips"] = (
                "PostgreSQL is offline or unreachable. Verify the server is running and the connection URL is correct."
            )

        else:
            diagnosis["status"] = "config_error"
            diagnosis["message"] = f"❌ Configuration error: {error_msg}"
            diagnosis["next_steps"] = [
                "1. Verify .streamlit/secrets.toml exists",
                "2. Check file has correct sections: [general], [db]",
                "3. Verify all required keys: api_key, db_url, vecdb_url, vecdb_collection, qdrant_api_key",
                "4. Check TOML syntax (no missing quotes or colons)"
            ]
            diagnosis["troubleshooting_tips"] = (
                "Configuration issue detected. Ensure .streamlit/secrets.toml is properly formatted with all required keys."
            )

    except FileNotFoundError as e:
        diagnosis["status"] = "config_missing"
        diagnosis["message"] = "❌ Configuration file NOT FOUND"
        diagnosis["next_steps"] = [
            "1. Create .streamlit directory: mkdir -p .streamlit",
            "2. Create secrets.toml: cat > .streamlit/secrets.toml << 'EOF'",
            "[general]",
            "api_key = \"sk-...\"",
            "model = \"gpt-4o-mini\"",
            "embedding_model = \"text-embedding-3-small\"",
            "",
            "[db]",
            "db_url = \"postgresql://pt:password@212.132.98.160:5432/pt_db\"",
            "vecdb_url = \"https://kasioss.com:6333\"",
            "vecdb_collection = \"pt_cv\"",
            "qdrant_api_key = \"...\"",
            "EOF"
        ]
        diagnosis["troubleshooting_tips"] = (
            f"Missing configuration file at {str(e)}. "
            "Create .streamlit/secrets.toml with all required database credentials."
        )

    except Exception as e:
        diagnosis["status"] = "unknown_error"
        diagnosis["message"] = f"❌ Unexpected error: {str(e)}"
        diagnosis["next_steps"] = [
            "1. Check error details above",
            "2. Review logs for detailed traceback",
            "3. Verify database connectivity: psql -c \"SELECT 1;\"",
            "4. Check file permissions on .streamlit/secrets.toml"
        ]
        diagnosis["troubleshooting_tips"] = (
            "An unexpected error occurred. Check the logs above for details."
        )

    return diagnosis


# ============================================================================
# DATABASE TOOLS
# ============================================================================

class DatabaseTools:
    """MCP Tools for accessing CV database"""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize DatabaseTools with configuration.

        Args:
            config: ConfigManager instance (optional, defaults to get_config())
        """
        self.config = config or get_config()
        self.pg_manager = get_postgres_manager()
        self.qdrant_manager = get_qdrant_manager()

        try:
            self.embedding_model = OpenAIEmbeddings(
                api_key=self.config.get_api_key(),
                model="text-embedding-3-small"
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise QdrantConnectionError(f"Failed to initialize embedding model: {e}")

        logger.info("DatabaseTools initialized with centralized managers")


    # ========================================================================
    # TOOL 1: Get CV Summary
    # ========================================================================
    def get_cv_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the CV including name, role, experience, and key stats.

        Returns:
            Dict with CV summary information
        """
        try:
            result = self.pg_manager.fetch_one("""
                SELECT name, crole as current_role, total_years_experience,
                       total_jobs, total_degrees, total_publications,
                       domains, all_skills
                FROM cv_summary
                LIMIT 1
            """)

            if result:
                logger.info("CV summary retrieved successfully")
                return {
                    "status": "success",
                    "tool": "get_cv_summary",
                    "summary": result
                }
            else:
                logger.warning("CV not found in database")
                return {
                    "status": "error",
                    "tool": "get_cv_summary",
                    "error": "CV not found"
                }

        except PostgreSQLConnectionError as e:
            logger.error(f"Database connection error in get_cv_summary: {e}")
            return {
                "status": "error",
                "tool": "get_cv_summary",
                "error": f"Database connection failed: {str(e)}"
            }
        except DatabaseQueryError as e:
            logger.error(f"Query error in get_cv_summary: {e}")
            return {
                "status": "error",
                "tool": "get_cv_summary",
                "error": f"Query failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error in get_cv_summary: {e}")
            return {
                "status": "error",
                "tool": "get_cv_summary",
                "error": str(e)
            }

    # ========================================================================
    # TOOL 2: Search Company Experience
    # ========================================================================
    def search_company_experience(self, company_name: str) -> Dict[str, Any]:
        """
        Find all jobs at a specific company.

        Args:
            company_name: Name of the company (e.g., 'KasiOss')

        Returns:
            Dict with work experience records
        """
        try:
            cv_id = self.config.get_cv_id()
            results = self.pg_manager.fetch_all("""
                SELECT company, role, location, start_date, end_date, is_current,
                       technologies, skills, domain, seniority, team_size
                FROM work_experience
                WHERE cv_id = %s AND company ILIKE %s
                ORDER BY start_date DESC
            """, (cv_id, f"%{company_name}%"))

            # Convert dates to strings
            for result in results:
                for key in ["start_date", "end_date"]:
                    if result.get(key):
                        result[key] = str(result[key])

            logger.info(f"Found {len(results)} jobs at {company_name}")
            return {
                "status": "success",
                "tool": "search_company_experience",
                "company": company_name,
                "results_count": len(results),
                "results": results
            }

        except CVNotFoundError as e:
            logger.error(f"CV not found in search_company_experience: {e}")
            return {
                "status": "error",
                "tool": "search_company_experience",
                "error": f"CV not found: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error in search_company_experience: {e}")
            return {
                "status": "error",
                "tool": "search_company_experience",
                "error": str(e)
            }

    # ========================================================================
    # TOOL 3: Search Technology Experience
    # ========================================================================
    def search_technology_experience(self, technology: str) -> Dict[str, Any]:
        """
        Find all jobs using a specific technology.

        Args:
            technology: Technology name (e.g., 'Python', 'TensorFlow')

        Returns:
            Dict with work experience using the technology
        """
        try:
            cv_id = self.config.get_cv_id()
            results = self.pg_manager.fetch_all("""
                SELECT company, role, start_date, end_date, technologies, domain
                FROM work_experience
                WHERE cv_id = %s AND %s = ANY(technologies)
                ORDER BY start_date DESC
            """, (cv_id, technology))

            # Convert dates
            for result in results:
                for key in ["start_date", "end_date"]:
                    if result.get(key):
                        result[key] = str(result[key])

            logger.info(f"Found {len(results)} jobs using {technology}")
            return {
                "status": "success",
                "tool": "search_technology_experience",
                "technology": technology,
                "results_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error in search_technology_experience: {e}")
            return {
                "status": "error",
                "tool": "search_technology_experience",
                "error": str(e)
            }

    # ========================================================================
    # TOOL 4: Search Work by Date Range
    # ========================================================================
    def search_work_by_date(self, start_year: int, end_year: int) -> Dict[str, Any]:
        """
        Find work experience within a date range.

        Args:
            start_year: Start year (e.g., 2020)
            end_year: End year (e.g., 2024)

        Returns:
            Dict with work experience in the date range
        """
        try:
            cv_id = self.config.get_cv_id()
            results = self.pg_manager.fetch_all("""
                SELECT company, role, start_date, end_date, technologies, keywords
                FROM work_experience
                WHERE cv_id = %s
                  AND start_date >= %s::date
                  AND (end_date <= %s::date OR end_date IS NULL)
                ORDER BY start_date DESC
            """, (cv_id, f"{start_year}-01-01", f"{end_year}-12-31"))

            for result in results:
                for key in ["start_date", "end_date"]:
                    if result.get(key):
                        result[key] = str(result[key])

            logger.info(f"Found {len(results)} jobs between {start_year}-{end_year}")
            return {
                "status": "success",
                "tool": "search_work_by_date",
                "date_range": f"{start_year}-{end_year}",
                "results_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error in search_work_by_date: {e}")
            return {
                "status": "error",
                "tool": "search_work_by_date",
                "error": str(e)
            }

    # ========================================================================
    # TOOL 5: Search Education
    # ========================================================================
    def search_education(self, institution: Optional[str] = None, degree: Optional[str] = None) -> Dict[str, Any]:
        """
        Find education records by institution or degree.

        Args:
            institution: University or institution name (optional)
            degree: Degree type (e.g., 'PhD', 'Master') (optional)

        Returns:
            Dict with education records
        """
        try:
            cv_id = self.config.get_cv_id()

            if institution:
                results = self.pg_manager.fetch_all("""
                    SELECT institution, degree, field, specialization, graduation_date, thesis, publications
                    FROM education
                    WHERE cv_id = %s AND institution ILIKE %s
                """, (cv_id, f"%{institution}%"))
                search_type = f"institution: {institution}"

            elif degree:
                results = self.pg_manager.fetch_all("""
                    SELECT institution, degree, field, specialization, graduation_date, thesis
                    FROM education
                    WHERE cv_id = %s AND degree ILIKE %s
                """, (cv_id, f"%{degree}%"))
                search_type = f"degree: {degree}"

            else:
                results = self.pg_manager.fetch_all("""
                    SELECT institution, degree, field, specialization, graduation_date, thesis
                    FROM education
                    WHERE cv_id = %s
                """, (cv_id,))
                search_type = "all education"

            for result in results:
                if result.get("graduation_date"):
                    result["graduation_date"] = str(result["graduation_date"])

            logger.info(f"Found {len(results)} education records for {search_type}")
            return {
                "status": "success",
                "tool": "search_education",
                "search_type": search_type,
                "results_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error in search_education: {e}")
            return {
                "status": "error",
                "tool": "search_education",
                "error": str(e)
            }

    # ========================================================================
    # TOOL 6: Search Publications
    # ========================================================================
    def search_publications(self, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Search publications by year.

        Args:
            year: Publication year (optional, defaults to all publications)

        Returns:
            Dict with publications
        """
        try:
            cv_id = self.config.get_cv_id()

            if year:
                results = self.pg_manager.fetch_all("""
                    SELECT title, year, conference_name, doi, keywords, content_text
                    FROM publications
                    WHERE cv_id = %s AND year = %s
                    ORDER BY year DESC
                """, (cv_id, year))
                search_type = f"year: {year}"

            else:
                results = self.pg_manager.fetch_all("""
                    SELECT title, year, conference_name, doi, keywords, content_text
                    FROM publications
                    WHERE cv_id = %s
                    ORDER BY year DESC
                """, (cv_id,))
                search_type = "all publications"

            logger.info(f"Found {len(results)} publications for {search_type}")
            return {
                "status": "success",
                "tool": "search_publications",
                "search_type": search_type,
                "results_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error in search_publications: {e}")
            return {
                "status": "error",
                "tool": "search_publications",
                "error": str(e)
            }

    # ========================================================================
    # TOOL 7: Search Skills
    # ========================================================================
    def search_skills(self, category: str) -> Dict[str, Any]:
        """
        Find skills by category.

        Args:
            category: Skill category (e.g., 'AI', 'ML', 'programming', 'Tools', 'Cloud', 'Data_tools')

        Returns:
            Dict with skills in the category
        """
        try:
            cv_id = self.config.get_cv_id()
            results = self.pg_manager.fetch_all("""
                SELECT skill_name
                FROM skills
                WHERE cv_id = %s AND skill_category = %s
                ORDER BY skill_name
            """, (cv_id, category))

            logger.info(f"Found {len(results)} skills in category {category}")
            return {
                "status": "success",
                "tool": "search_skills",
                "category": category,
                "results_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error in search_skills: {e}")
            return {
                "status": "error",
                "tool": "search_skills",
                "error": str(e)
            }

    # ========================================================================
    # TOOL 8: Search Awards and Certifications
    # ========================================================================
    def search_awards_certifications(self, award_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Find awards and certifications records.

        Args:
            award_type: Type of award/certification to filter (optional)

        Returns:
            Dict with awards and certifications records
        """
        try:
            cv_id = self.config.get_cv_id()

            if award_type:
                results = self.pg_manager.fetch_all("""
                    SELECT title, issuing_organization, organization, issue_date, keywords
                    FROM awards_certifications
                    WHERE cv_id = %s AND (issuing_organization ILIKE %s OR organization ILIKE %s OR title ILIKE %s)
                    ORDER BY issue_date DESC
                """, (cv_id, f"%{award_type}%", f"%{award_type}%", f"%{award_type}%"))
                search_type = f"type: {award_type}"

            else:
                results = self.pg_manager.fetch_all("""
                    SELECT title, issuing_organization, organization, issue_date, keywords
                    FROM awards_certifications
                    WHERE cv_id = %s
                    ORDER BY issue_date DESC
                """, (cv_id,))
                search_type = "all awards and certifications"

            for result in results:
                if result.get("issue_date"):
                    result["issue_date"] = str(result["issue_date"])

            logger.info(f"Found {len(results)} awards/certifications for {search_type}")
            return {
                "status": "success",
                "tool": "search_awards_certifications",
                "search_type": search_type,
                "results_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error in search_awards_certifications: {e}")
            return {
                "status": "error",
                "tool": "search_awards_certifications",
                "error": str(e)
            }

    # ========================================================================
    # TOOL 9: Semantic Search
    # ========================================================================
    def semantic_search(self, query: str, section: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform semantic search on CV content using vector embeddings.

        Args:
            query: Natural language search query
            section: Filter by section (work_experience, education, publication, all)
            top_k: Number of results to return (default: 5)

        Returns:
            Dict with semantic search results from Qdrant
        """
        try:
            query_embedding = self.embedding_model.embed_query(query)

            search_params = {
                "collection_name": self.config.get_qdrant_collection(),
                "query_vector": query_embedding,
                "limit": top_k
            }

            # Add section filter if provided
            if section and section != "all":
                
                search_params["query_filter"] = Filter(
                    must=[
                        FieldCondition(
                            key="section",
                            match=MatchValue(value=section)
                        )
                    ]
                )

            results = self.qdrant_manager.client.search(**search_params)

            formatted_results = []
            for result in results:
                # Build result with core fields
                formatted_result = {
                    "chunk_id": result.payload.get("chunk_id"),
                    "cv_id": result.payload.get("cv_id"),
                    "section": result.payload.get("section"),
                    "similarity_score": result.score
                }

                # Add section-specific metadata fields
                section_value = result.payload.get("section", "")

                # Work experience specific fields
                if section_value == "work experience":
                    if result.payload.get("company"):
                        formatted_result["company"] = result.payload.get("company")
                    if result.payload.get("role"):
                        formatted_result["role"] = result.payload.get("role")
                    if result.payload.get("domain"):
                        formatted_result["domain"] = result.payload.get("domain")
                    if result.payload.get("responsibility"):
                        formatted_result["responsibility"] = result.payload.get("responsibility")

                # Education specific fields
                elif section_value == "education":
                    if result.payload.get("institution"):
                        formatted_result["institution"] = result.payload.get("institution")
                    if result.payload.get("degree"):
                        formatted_result["degree"] = result.payload.get("degree")
                    if result.payload.get("thesis"):
                        formatted_result["thesis"] = result.payload.get("thesis")
                    if result.payload.get("graduation_date"):
                        formatted_result["graduation_date"] = result.payload.get("graduation_date")

                # Publication specific fields
                elif section_value == "publication":
                    if result.payload.get("title"):
                        formatted_result["title"] = result.payload.get("title")

                # Projects specific fields
                elif section_value == "projects":
                    if result.payload.get("project_name"):
                        formatted_result["project_name"] = result.payload.get("project_name")
                    if result.payload.get("responsibility"):
                        formatted_result["responsibility"] = result.payload.get("responsibility")
                    if result.payload.get("technologies"):
                        formatted_result["technologies"] = result.payload.get("technologies")

                # Common optional fields
                if result.payload.get("technologies"):
                    formatted_result["technologies"] = result.payload.get("technologies")
                if result.payload.get("skills"):
                    formatted_result["skills"] = result.payload.get("skills")

                formatted_results.append(formatted_result)

            logger.info(f"Semantic search found {len(formatted_results)} results for query: '{query}'")
            return {
                "status": "success",
                "tool": "semantic_search",
                "query": query,
                "section_filter": section or "all",
                "results_count": len(formatted_results),
                "results": formatted_results
            }

        except Exception as e:
            logger.error(f"Error in semantic_search: {e}")
            return {
                "status": "error",
                "tool": "semantic_search",
                "error": str(e)
            }

    # ========================================================================
    # TOOL 10: Get All Work Experience (Complete Career History) ⭐ PRIMARY FOR EXPERIENCE QUERIES
    # ========================================================================
    def get_all_work_experience(self) -> Dict[str, Any]:
        """
        Get complete work experience history - all jobs in chronological order.

        ⭐ PRIMARY TOOL for general "experience" queries!
        This tool returns the ENTIRE work_experience table for the CV,
        perfect for questions about:
        - "Her experience?"
        - "Work history?" / "Career history?"
        - "List of jobs?" / "All jobs?"
        - "Career timeline?" / "Career background?"
        - "Where did she work?"

        Returns:
            Dict with all work experience records (complete career history) ordered by date DESC

        Response Format:
            {
                "status": "success",
                "tool": "get_all_work_experience",
                "results_count": int,
                "results": [
                    {
                        "company": str,
                        "role": str,
                        "location": str,
                        "start_date": str,
                        "end_date": str,
                        "is_current": bool,
                        "technologies": list,
                        "skills": list,
                        "domain": str,
                        "seniority": str,
                        "team_size": int
                    },
                    ...
                ]
            }
        """
        try:
            cv_id = self.config.get_cv_id()
            results = self.pg_manager.fetch_all("""
                SELECT company, role, location, start_date, end_date, is_current,
                       technologies, skills, domain, seniority, team_size
                FROM work_experience
                WHERE cv_id = %s
                ORDER BY start_date DESC
            """, (cv_id,))

            # Convert dates to strings
            for result in results:
                for key in ["start_date", "end_date"]:
                    if result.get(key):
                        result[key] = str(result[key])

            logger.info(f"Retrieved {len(results)} work experience records")
            return {
                "status": "success",
                "tool": "get_all_work_experience",
                "results_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error in get_all_work_experience: {e}")
            return {
                "status": "error",
                "tool": "get_all_work_experience",
                "error": str(e)
            }


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

def create_mcp_server():
    """Create and return configured MCP server"""
    try:
        config = get_config()
        tools = DatabaseTools(config)
        logger.info("MCP Server initialized successfully")
        return tools
    except MCPServerError as e:
        logger.error(f"MCP Server initialization error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MCP Server: {e}")
        raise MCPServerError(f"MCP Server initialization failed: {str(e)}")


if __name__ == "__main__":
    logger.info("MCP Server started")
    mcp_server = create_mcp_server()
    logger.info("All tools available and ready")
