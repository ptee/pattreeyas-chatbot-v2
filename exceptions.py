"""
Custom exceptions for CV LLM Agent
Provides structured error handling for configuration, database, and LLM operations
"""

from typing import List


class CVAgentException(Exception):
    """Base exception for all CV agent errors"""
    pass


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(CVAgentException):
    """Raised when configuration loading or validation fails"""
    pass


class ConfigurationMissingError(ConfigurationError):
    """Raised when configuration file or keys are missing"""
    pass


class ConfigurationInvalidError(ConfigurationError):
    """Raised when configuration values are invalid"""
    pass


class StreamlitSecretsError(ConfigurationError):
    """Raised when Streamlit secrets cannot be accessed"""
    pass


class EnvVarError(ConfigurationError):
    """Raised when required environment variables are missing"""
    pass


# ============================================================================
# DATABASE CONNECTION EXCEPTIONS
# ============================================================================

class DatabaseConnectionError(CVAgentException):
    """Raised when database connection fails"""
    pass


class PostgreSQLConnectionError(DatabaseConnectionError):
    """Raised when PostgreSQL connection fails"""
    pass


class QdrantConnectionError(DatabaseConnectionError):
    """Raised when Qdrant vector DB connection fails"""
    pass


class DatabaseOperationError(CVAgentException):
    """Raised when a database operation fails"""
    pass


class DatabaseQueryError(DatabaseOperationError):
    """Raised when a SQL query fails"""
    pass


class DatabaseInsertError(DatabaseOperationError):
    """Raised when data insertion fails"""
    pass


class DatabaseTableError(DatabaseOperationError):
    """Raised when table operations fail"""
    pass


# ============================================================================
# DATA VALIDATION EXCEPTIONS
# ============================================================================

class DataValidationError(CVAgentException):
    """Raised when data validation fails"""
    pass


class InvalidUUIDError(DataValidationError):
    """Raised when UUID format is invalid"""
    pass


class CVNotFoundError(DataValidationError):
    """Raised when CV data is not found in database"""
    pass


class InvalidDataFormatError(DataValidationError):
    """Raised when data format is invalid"""
    pass


# ============================================================================
# EMBEDDING AND VECTOR DB EXCEPTIONS
# ============================================================================

class EmbeddingError(CVAgentException):
    """Raised when text embedding fails"""
    pass


class VectorStoreError(CVAgentException):
    """Raised when vector store operations fail"""
    pass


class VectorSearchError(VectorStoreError):
    """Raised when semantic search fails"""
    pass


# ============================================================================
# LLM AGENT EXCEPTIONS
# ============================================================================

class LLMAgentError(CVAgentException):
    """Raised when LLM agent operations fail"""
    pass


class ToolCallingError(LLMAgentError):
    """Raised when LLM tool calling fails"""
    pass


class LLMResponseError(LLMAgentError):
    """Raised when LLM response processing fails"""
    pass


class NoToolsCalledError(ToolCallingError):
    """Raised when LLM should have called tools but didn't"""
    pass


# ============================================================================
# MCP (MODEL CONTEXT PROTOCOL) EXCEPTIONS
# ============================================================================

class MCPError(CVAgentException):
    """Raised when MCP server/client operations fail"""
    pass


class MCPClientError(MCPError):
    """Raised when MCP client initialization or operation fails"""
    pass


class MCPServerError(MCPError):
    """Raised when MCP server operation fails"""
    pass


class MCPToolError(MCPError):
    """Raised when MCP tool execution fails"""
    pass


# ============================================================================
# EXCEPTION CONTEXT HELPERS
# ============================================================================

class ErrorContext:
    """Helper for providing detailed error context"""

    def __init__(self, error_type: str, message: str, suggestions: List[str] = None,
                 remediation_steps: List[str] = None):
        """
        Initialize error context

        Args:
            error_type: Type of error (e.g., "config_missing", "db_connection_failed")
            message: Human-readable error message
            suggestions: List of suggestions to resolve the error
            remediation_steps: Step-by-step instructions to fix the error
        """
        self.error_type = error_type
        self.message = message
        self.suggestions = suggestions or []
        self.remediation_steps = remediation_steps or []

    def format(self) -> str:
        """Format error context as readable string"""
        lines = [f"\n{'='*70}"]
        lines.append(f"ERROR: {self.error_type.upper()}")
        lines.append(f"{'='*70}")
        lines.append(f"\nMessage: {self.message}\n")

        if self.suggestions:
            lines.append("💡 Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"  • {suggestion}")

        if self.remediation_steps:
            lines.append("\n🔧 Remediation Steps:")
            for i, step in enumerate(self.remediation_steps, 1):
                lines.append(f"  {i}. {step}")

        lines.append(f"\n{'='*70}\n")
        return "\n".join(lines)


# Convenience function for creating exceptions with context
def create_contextual_error(exception_class, error_context: ErrorContext) -> Exception:
    """
    Create an exception with formatted error context

    Args:
        exception_class: The exception class to instantiate
        error_context: ErrorContext instance with detailed information

    Returns:
        Exception instance with formatted message
    """
    return exception_class(error_context.format())