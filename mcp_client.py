"""
MCP Client for CV Database Access
Wrapper for MCP Server tools to be used by LLM agents

Refactored to use:
- ConfigManager (config.py) for unified configuration
- DatabaseTools (mcp_server.py) for database operations
- Custom exceptions (exceptions.py) for better error handling
"""

import logging
from typing import Dict, List, Any, Optional

from config import get_config, ConfigManager
from mcp_server import DatabaseTools
from exceptions import MCPServerError

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, "WARNING"))

class MCPClient:
    """Client for accessing MCP database tools"""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize MCP client.

        Args:
            config: ConfigManager instance (optional, defaults to get_config())

        Raises:
            MCPServerError: If initialization fails
        """
        try:
            self.config = config or get_config()
            self.tools = DatabaseTools(self.config)

            # Initialize CV ID at startup (fetches it once and caches it)
            # This ensures we fail early if database is not configured properly
            self.cv_id = self.config.get_cv_id()
            logger.info(f"MCP Client initialized successfully with CV ID: {self.cv_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to initialize MCP Client: {e}")
            raise MCPServerError(f"MCP Client initialization failed: {str(e)}")

    # ========================================================================
    # Tool 1: Get CV Summary
    # ========================================================================
    def get_cv_summary(self) -> Dict[str, Any]:
        """Get a summary of the CV including role, experience, and key stats"""
        return self.tools.get_cv_summary()

    # ========================================================================
    # Tool 2: Search Company Experience
    # ========================================================================
    def search_company_experience(self, company_name: str) -> Dict[str, Any]:
        """Find all jobs at a specific company"""
        return self.tools.search_company_experience(company_name)

    # ========================================================================
    # Tool 3: Search Technology Experience
    # ========================================================================
    def search_technology_experience(self, technology: str) -> Dict[str, Any]:
        """Find all jobs using a specific technology"""
        return self.tools.search_technology_experience(technology)

    # ========================================================================
    # Tool 4: Search Work by Date
    # ========================================================================
    def search_work_by_date(self, start_year: int, end_year: int) -> Dict[str, Any]:
        """Find work experience within a date range"""
        return self.tools.search_work_by_date(start_year, end_year)

    # ========================================================================
    # Tool 5: Search Education
    # ========================================================================
    def search_education(self, institution: Optional[str] = None, degree: Optional[str] = None) -> Dict[str, Any]:
        """Find education records by institution or degree"""
        return self.tools.search_education(institution, degree)

    # ========================================================================
    # Tool 6: Search Publications
    # ========================================================================
    def search_publications(self, year: Optional[int] = None) -> Dict[str, Any]:
        """Search publications by year or get all publications"""
        return self.tools.search_publications(year)

    # ========================================================================
    # Tool 7: Search Skills
    # ========================================================================
    def search_skills(self, category: str) -> Dict[str, Any]:
        """Find skills by category"""
        return self.tools.search_skills(category)

    # ========================================================================
    # Tool 8: Search Awards and Certifications
    # ========================================================================
    def search_awards_certifications(self, award_type: Optional[str] = None) -> Dict[str, Any]:
        """Find awards and certifications records"""
        return self.tools.search_awards_certifications(award_type)

    # ========================================================================
    # Tool 9: Semantic Search
    # ========================================================================
    def semantic_search(self, query: str, section: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Perform semantic search on CV content using vector embeddings"""
        return self.tools.semantic_search(query, section, top_k)

    # ========================================================================
    # Tool 10: Get All Work Experience ⭐ PRIMARY FOR EXPERIENCE QUERIES
    # ========================================================================
    def get_all_work_experience(self) -> Dict[str, Any]:
        """
        Get complete work experience history - all jobs in chronological order.

        ⭐ PRIMARY TOOL for general "experience" queries!
        Returns ENTIRE work history at once, perfect for:
        - "Her experience?"
        - "Work history?" or "Career history?"
        - "All jobs?" or "List of experience?"
        - "Career timeline?" or "Work background?"

        Returns:
            Dict with all work records including company, role, location, dates,
            technologies, skills, domain, seniority, and team size
        """
        return self.tools.get_all_work_experience()

    # ========================================================================
    # Tool Registry
    # ========================================================================
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available MCP tools"""
        return [
            {
                "name": "get_cv_summary",
                "description": "Get a summary of the CV including role, experience, and key stats",
                "parameters": {}
            },
            {
                "name": "search_company_experience",
                "description": "Find all jobs at a specific company",
                "parameters": {
                    "company_name": {"type": "string", "description": "Name of the company"}
                }
            },
            {
                "name": "search_technology_experience",
                "description": "Find all jobs using a specific technology",
                "parameters": {
                    "technology": {"type": "string", "description": "Technology name"}
                }
            },
            {
                "name": "search_work_by_date",
                "description": "Find work experience within a date range",
                "parameters": {
                    "start_year": {"type": "integer", "description": "Start year"},
                    "end_year": {"type": "integer", "description": "End year"}
                }
            },
            {
                "name": "search_education",
                "description": "Find education records by institution or degree",
                "parameters": {
                    "institution": {"type": "string", "description": "Institution name (optional)"},
                    "degree": {"type": "string", "description": "Degree type (optional)"}
                }
            },
            {
                "name": "search_publications",
                "description": "Search publications by year",
                "parameters": {
                    "year": {"type": "integer", "description": "Publication year (optional)"}
                }
            },
            {
                "name": "search_skills",
                "description": "Find skills by category",
                "parameters": {
                    "category": {"type": "string", "description": "Skill category"}
                }
            },
            {
                "name": "search_awards_certifications",
                "description": "Find awards and certifications records",
                "parameters": {
                    "award_type": {"type": "string", "description": "Award or certification type (optional)"}
                }
            },
            {
                "name": "semantic_search",
                "description": "Perform semantic search on CV content using vector embeddings",
                "parameters": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "section": {"type": "string", "description": "Filter by section (optional)"},
                    "top_k": {"type": "integer", "description": "Number of results (default: 5)"}
                }
            },
            {
                "name": "get_all_work_experience",
                "description": "⭐ PRIMARY TOOL: Get COMPLETE work experience history in chronological order - ALL jobs at once. MUST USE for any 'experience', 'work history', 'career history', 'jobs', 'career timeline', 'work background' questions. Returns all work records with company, role, dates, technologies, skills, domain, seniority, team size.",
                "parameters": {}
            }
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with given parameters"""
        if tool_name == "get_cv_summary":
            return self.get_cv_summary()

        elif tool_name == "search_company_experience":
            return self.search_company_experience(kwargs.get("company_name", ""))

        elif tool_name == "search_technology_experience":
            return self.search_technology_experience(kwargs.get("technology", ""))

        elif tool_name == "search_work_by_date":
            return self.search_work_by_date(
                kwargs.get("start_year", 2000),
                kwargs.get("end_year", 2024)
            )

        elif tool_name == "search_education":
            return self.search_education(
                institution=kwargs.get("institution"),
                degree=kwargs.get("degree")
            )

        elif tool_name == "search_publications":
            return self.search_publications(year=kwargs.get("year"))

        elif tool_name == "search_skills":
            return self.search_skills(kwargs.get("category", ""))

        elif tool_name == "search_awards_certifications":
            return self.search_awards_certifications(award_type=kwargs.get("award_type"))

        elif tool_name == "semantic_search":
            return self.semantic_search(
                query=kwargs.get("query", ""),
                section=kwargs.get("section"),
                top_k=kwargs.get("top_k", 5)
            )

        elif tool_name == "get_all_work_experience":
            return self.get_all_work_experience()

        else:
            return {
                "status": "error",
                "tool": tool_name,
                "error": f"Unknown tool: {tool_name}"
            }


# Global client instance
_client = None


def get_mcp_client() -> MCPClient:
    """Get or create global MCP client instance"""
    global _client
    if _client is None:
        _client = MCPClient()
    return _client


if __name__ == "__main__":
    # Test the client
    client = get_mcp_client()
    print(f"Available tools: {len(client.get_available_tools())}")

    # Test a simple tool
    result = client.get_cv_summary()
    print(f"CV Summary: {result['status']}")
