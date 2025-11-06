"""
Centralized Configuration Management for CV LLM Agent
Handles configuration loading from Streamlit secrets, TOML files, and environment variables
Provides single source of truth for all application configuration
"""

import os
import logging
import toml
from typing import Dict, Any, Optional
from uuid import UUID
import psycopg2

import streamlit as st

from exceptions import (
    ConfigurationError,
    ConfigurationMissingError,
    ConfigurationInvalidError,
    StreamlitSecretsError,
    EnvVarError,
    PostgreSQLConnectionError,
    InvalidUUIDError,
    CVNotFoundError
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION MANAGER CLASS
# ============================================================================

class ConfigManager:
    """
    Centralized configuration manager for CV LLM Agent

    Loads configuration from multiple sources in priority order:
    1. Streamlit secrets (if running in Streamlit context)
    2. .streamlit/secrets.toml file
    3. Environment variables (fallback)

    Provides cached access to configuration values and CV ID validation.
    """

    def __init__(self, skip_cv_id_validation: bool = False):
        """
        Initialize ConfigManager

        Args:
            skip_cv_id_validation: If True, skip fetching CV ID on init (for non-agent contexts)

        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        self._config = {}
        self._cv_id = None
        self._cv_id_loaded = False
        self._is_streamlit = self._detect_streamlit_context()

        try:
            if self._is_streamlit:
                self._load_from_streamlit_secrets()
            else:
                self._load_from_toml_or_env()

            logger.info("✓ Configuration loaded successfully")

            # Validate required keys
            self._validate_config()

            # Optionally fetch and cache CV ID
            if not skip_cv_id_validation:
                _ = self.get_cv_id()

        except Exception as e:
            logger.error(f"Failed to initialize ConfigManager: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}") from e

    # ========================================================================
    # CONFIGURATION LOADING METHODS
    # ========================================================================

    def _detect_streamlit_context(self) -> bool:
        """Detect if running in Streamlit context"""
        try:
            _ = st.runtime.exists()
            return True
        except (AttributeError, RuntimeError):
            return False

    def _load_from_streamlit_secrets(self) -> None:
        """Load configuration from Streamlit secrets"""
        try:
            self._config = {
                "general": {
                    "api_key": st.secrets["general"]["api_key"],
                    "model": st.secrets["general"]["model"],
                    "embedding_model": st.secrets["general"]["embedding_model"],
                },
                "db": {
                    "db_url": st.secrets["db"]["db_url"],
                    "vecdb_url": st.secrets["db"]["vecdb_url"],
                    "vecdb_collection": st.secrets["db"]["vecdb_collection"],
                    "qdrant_api_key": st.secrets["db"]["qdrant_api_key"],
                }
            }
            logger.info("Configuration loaded from Streamlit secrets")
        except (KeyError, StreamlitSecretsError) as e:
            logger.error(f"Missing Streamlit secret: {e}")
            raise StreamlitSecretsError(f"Streamlit secret missing: {e}") from e

    def _load_from_toml_or_env(self) -> None:
        """Load configuration from TOML file or environment variables"""
        secrets_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".streamlit/secrets.toml"
        )

        # Try TOML first
        if os.path.exists(secrets_path):
            try:
                with open(secrets_path, "r") as f:
                    self._config = toml.load(f)
                logger.info(f"Configuration loaded from {secrets_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load TOML: {e}, falling back to env vars")
        # Fallback to environment variables
        # Don't allow for default values here; all must be set
        
        logger.info("Configuration loaded from environment variables")

    def _validate_config(self) -> None:
        """Validate that all required configuration keys exist"""
        required_keys = {
            "general": ["api_key", "model", "embedding_model"],
            "db": ["db_url", "vecdb_url", "vecdb_collection", "qdrant_api_key"]
        }

        for section, keys in required_keys.items():
            if section not in self._config:
                raise ConfigurationMissingError(f"Missing section: [{section}]")

            for key in keys:
                if key not in self._config[section]:
                    raise ConfigurationMissingError(f"Missing key: {section}.{key}")

                value = self._config[section][key]
                if value is None or (isinstance(value, str) and not value.strip()):
                    if key == "qdrant_api_key":  # Optional in some contexts
                        logger.warning(f"Configuration key is empty: {section}.{key}")
                    else:
                        raise ConfigurationInvalidError(f"Configuration key is empty: {section}.{key}")

    # ========================================================================
    # CONFIGURATION GETTERS
    # ========================================================================

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value

        Args:
            section: Configuration section (e.g., "general", "db")
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(section, {}).get(key, default)

    def get_api_key(self) -> str:
        """Get OpenAI API key"""
        return self.get("general", "api_key")

    def get_model(self) -> str:
        """Get LLM model name"""
        return self.get("general", "model", "gpt-4o-mini")

    def get_embedding_model(self) -> str:
        """Get embedding model name"""
        return self.get("general", "embedding_model", "text-embedding-3-small")

    def get_postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return self.get("db", "db_url")

    def get_qdrant_url(self) -> str:
        """Get Qdrant vector DB URL"""
        return self.get("db", "vecdb_url")

    def get_qdrant_collection(self) -> str:
        """Get Qdrant collection name"""
        return self.get("db", "vecdb_collection", "pt_cv")

    def get_qdrant_api_key(self) -> str:
        """Get Qdrant API key"""
        return self.get("db", "qdrant_api_key", "")

    # ========================================================================
    # CV ID MANAGEMENT
    # ========================================================================

    def get_cv_id(self) -> str:
        """
        Get the CV ID from the database (cached after first retrieval)

        Returns:
            str: Valid UUID string of the first CV in the database

        Raises:
            PostgreSQLConnectionError: If database connection fails
            InvalidUUIDError: If CV ID is not a valid UUID
            CVNotFoundError: If no CV data found in database
        """
        # Return cached CV ID if already loaded
        if self._cv_id_loaded:
            if self._cv_id is None:
                raise CVNotFoundError("No CV data found in database. Please run db_ingestion.py to load data.")
            return self._cv_id

        try:
            conn = psycopg2.connect(self.get_postgres_url())
            cursor = conn.cursor()

            # Query CV ID from cv_metadata table
            cursor.execute("SELECT id FROM cv_metadata LIMIT 1")
            result = cursor.fetchone()

            cursor.close()
            conn.close()

            if not result:
                self._cv_id_loaded = True
                self._cv_id = None
                raise CVNotFoundError("No CV data found in database. Please run db_ingestion.py to load data.")

            # Validate UUID format
            cv_id = str(result[0])
            try:
                UUID(cv_id)  # This will raise ValueError if not a valid UUID
                self._cv_id = cv_id
                self._cv_id_loaded = True
                logger.info(f"✓ Found and cached CV ID: {cv_id[:8]}...")
                return cv_id
            except ValueError as e:
                self._cv_id_loaded = True
                raise InvalidUUIDError(f"CV ID in database is not a valid UUID: {cv_id}") from e

        except psycopg2.Error as e:
            self._cv_id_loaded = True
            logger.error(f"Database error while fetching CV ID: {e}")
            raise PostgreSQLConnectionError(f"Failed to fetch CV ID from database: {e}") from e
        except (CVNotFoundError, InvalidUUIDError):
            # Re-raise our custom errors
            raise
        except Exception as e:
            self._cv_id_loaded = True
            logger.error(f"Unexpected error while fetching CV ID: {e}")
            raise ConfigurationError(f"Unexpected error fetching CV ID: {e}") from e

    def clear_cv_id_cache(self) -> None:
        """Clear the cached CV ID (useful for testing)"""
        self._cv_id = None
        self._cv_id_loaded = False
        logger.info("Cleared CV ID cache")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return self._config.copy()

    def is_streamlit_context(self) -> bool:
        """Check if running in Streamlit context"""
        return self._is_streamlit

    def __repr__(self) -> str:
        """String representation of config (masks sensitive values)"""
        masked = {
            "general": {
                "api_key": "***MASKED***",
                "model": self.get_model(),
                "embedding_model": self.get_embedding_model(),
            },
            "db": {
                "db_url": "***MASKED***",
                "vecdb_url": self.get_qdrant_url(),
                "vecdb_collection": self.get_qdrant_collection(),
                "qdrant_api_key": "***MASKED***",
            }
        }
        return f"ConfigManager({masked})"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_config_instance: Optional[ConfigManager] = None


def get_config(skip_cv_id_validation: bool = False) -> ConfigManager:
    """
    Get or create the global ConfigManager instance (singleton pattern)

    Args:
        skip_cv_id_validation: If True, skip CV ID validation on initialization

    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(skip_cv_id_validation=skip_cv_id_validation)
    return _config_instance


def reset_config() -> None:
    """Reset the global ConfigManager instance (useful for testing)"""
    global _config_instance
    _config_instance = None
    logger.info("Reset global ConfigManager instance")