"""
Semantic utilities for improved question classification and similarity tracking.
Implements Solution 1 (Category Classification) and Solution 5 (Similarity Tracking).

REFACTORED to use ConfigManager for unified configuration management.
All LLM interactions now go through ConfigManager for API key and model selection.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_config, ConfigManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Set semantic_utils logging to WARNING level


# ============================================================================
# LANGUAGE DETECTION UTILITY
# ============================================================================

def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of the input text.

    Uses langid library for fast, lightweight language detection.
    Falls back to English if detection fails.

    Args:
        text: Text to analyze for language

    Returns:
        Tuple of (language_code, confidence)
        - language_code: ISO 639-1 language code (e.g., 'en', 'de', 'fr', 'th')
        - confidence: Confidence score (0-1)
    """
    if not text or len(text.strip()) < 3:
        logger.debug("Text too short for language detection, defaulting to English")
        return "en", 0.0

    try:
        import langid
        detected_lang, confidence = langid.classify(text)
        logger.debug(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")
        return detected_lang, confidence
    except ImportError:
        logger.warning("langid not installed, defaulting to English. Install with: pip install langid")
        return "en", 0.0
    except Exception as e:
        logger.warning(f"Language detection failed: {e}, defaulting to English")
        return "en", 0.0


def translate_text(text: str, target_language: str) -> str:
    """
    Translate text to target language using translate library.

    Uses the `translate` library with Translator for quick, offline translation.
    Falls back to original text if translation fails.

    Args:
        text: Text to translate
        target_language: ISO 639-1 language code (e.g., 'de', 'fr', 'th')

    Returns:
        Translated text, or original text if translation fails
    """
    # Don't translate if target is English
    if target_language == "en" or not text:
        return text

    try:
        from translate import Translator
        translator = Translator(from_lang="en", to_lang=target_language)
        translated = translator.translate(text)
        logger.debug(f"Translated to {target_language}: {text[:50]}... → {translated[:50]}...")
        return translated
    except ImportError:
        logger.warning("translate library not installed, returning original text. Install with: pip install translate")
        return text
    except Exception as e:
        logger.warning(f"Translation to {target_language} failed: {e}, returning original text")
        return text


# ============================================================================
# SOLUTION 1: Semantic Category Classification
# ============================================================================

class QuestionCategoryClassifier:
    """
    Classifies CV-related questions into 7 categories using LLM semantic understanding.
    Replaces keyword-based matching for accurate category detection.

    Categories:
    1. General Overview - Who is Pattreeya? Summary of background
    2. Work Experience - Roles, companies, responsibilities
    3. Technical Skills - Programming languages, frameworks, AI/ML expertise
    4. Education - Degrees, universities, thesis
    5. Publications - Papers, research, articles
    6. Awards & Certifications - Awards, honors, certifications
    7. Comprehensive - Deep learning, languages, broader skills
    """

    def __init__(self, config: Optional[ConfigManager] = None, temperature: float = 0.0) -> None:
        """
        Initialize the classifier with API config.

        Args:
            config: ConfigManager with API credentials (optional, defaults to get_config())
            temperature: LLM temperature (0.0 for deterministic classification)

        Raises:
            ConfigurationError: If API credentials cannot be loaded
        """
        self.config = config or get_config()
        self.llm = ChatOpenAI(
            api_key=self.config.get_api_key(),
            model=self.config.get_model(),
            temperature=temperature,  # Deterministic for consistent classification
            max_tokens=50
        )
        self.cache = {}  # Cache classifications to avoid re-classifying same questions
        self.valid_categories = [
            "General Overview",
            "Work Experience",
            "Technical Skills",
            "Education",
            "Publications",
            "Awards & Certifications",
            "Comprehensive"
        ]

    def classify_question(self, question: str) -> str:
        """
        Classify a question into one of 7 categories using LLM.
        Uses caching to prevent redundant API calls.

        Args:
            question: User's question about Pattreeya

        Returns:
            One of the 7 valid category names
        """

        # Check cache first
        question_hash = hash(question.lower())
        if question_hash in self.cache:
            cached_result = self.cache[question_hash]
            logger.debug(f"Cache hit for: '{question[:50]}...' → {cached_result}")
            return cached_result

        try:
            from prompts import CATEGORY_CLASSIFIER_PROMPT

            messages = [
                SystemMessage(content=CATEGORY_CLASSIFIER_PROMPT),
                HumanMessage(content=f"Question: {question}")
            ]

            response = self.llm.invoke(messages)
            category = response.content.strip()

            # Validate category is in valid list
            if category not in self.valid_categories:
                # Try to extract category name from response
                for valid_cat in self.valid_categories:
                    if valid_cat.lower() in category.lower():
                        category = valid_cat
                        break
                else:
                    # Fallback to General Overview if unrecognized
                    logger.warning(f"Unrecognized category '{category}'. Defaulting to 'General Overview'")
                    category = "General Overview"

            # Cache the result
            self.cache[question_hash] = category
            logger.info(f"Classified: '{question[:60]}...' → {category}")
            return category

        except Exception as e:
            logger.error(f"Error classifying question: {e}. Defaulting to 'General Overview'")
            return "General Overview"

    def clear_cache(self):
        """Clear the classification cache."""
        self.cache.clear()
        logger.info("Classifier cache cleared")


# ============================================================================
# FOLLOW-UP QUESTION GENERATION (Refactored from llm-agent.py)
# ============================================================================

class FollowUpQuestionGenerator:
    """
    Handles all follow-up question generation logic including:
    - Category classification
    - Category rotation with priority mapping
    - Awards triggering with cooling-off periods
    - Question generation using pre-generated questions (fast path)

    This encapsulates the follow-up logic from final_response_node
    to improve code organization and reusability.
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the follow-up question generator.

        Args:
            config: ConfigManager with API credentials (optional, defaults to get_config())
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)

        # Category definitions
        self.all_categories = [
            "General Overview",
            "Work Experience",
            "Technical Skills",
            "Education",
            "Publications",
            "Awards & Certifications",
            "Comprehensive"
        ]

        # Category priority transitions (current → [priority_list])
        self.category_priority_map = {
            "Technical Skills": ["Education", "Work Experience", "Publications"],
            "Work Experience": ["Publications", "Education", "Technical Skills"],
            "Education": ["Work Experience", "Publications", "Technical Skills"],
            "Publications": ["Technical Skills", "Work Experience", "Education"],
            "Awards & Certifications": ["Education", "Publications", "Work Experience"],
            "General Overview": ["Work Experience", "Education", "Technical Skills"],
            "Comprehensive": ["Work Experience", "Education", "Publications"]
        }

    def get_recommended_category(
        self,
        current_category: str,
        explored_categories: List[str],
        conversation_history: List[Dict],
        user_query: str
    ) -> str:
        """
        Determine the recommended next category for follow-up question.

        Implements:
        - SOLUTION 2: Smart awards triggering with cooling-off periods
        - SOLUTION 3: Category rotation with priority filtering

        Args:
            current_category: Current question's category
            explored_categories: List of categories already explored
            conversation_history: History of previous questions
            user_query: Current user query for keyword matching

        Returns:
            Recommended category for next question
        """
        # SOLUTION 2: Awards triggering logic
        current_query_lower = user_query.lower()
        awards_trigger_keywords = [
            "award", "awards", "certification", "certifications", "certified",
            "recognition", "recognitions", "recognized",
            "honor", "honors", "honoured"
        ]
        should_trigger_awards = any(kw in current_query_lower for kw in awards_trigger_keywords)
        awards_already_covered = "Awards & Certifications" in explored_categories

        # SOLUTION 3: Priority filtering with category rotation
        base_recommended_priorities = self.category_priority_map.get(current_category, self.all_categories[1:])

        # Use base priorities for rotation (no semantic tracking needed)
        recommended_priorities = base_recommended_priorities

        # Awards promotion
        if should_trigger_awards and not awards_already_covered and "Awards & Certifications" not in recommended_priorities:
            recommended_priorities = ["Awards & Certifications"] + recommended_priorities

        # Find first unexplored category from priority list
        for cat in recommended_priorities:
            if cat not in explored_categories:
                return cat

        # If all priorities explored, find least-mentioned category
        category_mention_count = {cat: explored_categories.count(cat) for cat in self.all_categories}
        unexplored = [cat for cat in self.all_categories if cat not in explored_categories]

        if unexplored:
            return unexplored[0]

        # All explored - return least-mentioned (prefer non-awards)
        least_mentioned = min(category_mention_count, key=category_mention_count.get)
        if least_mentioned == "Awards & Certifications" and len(explored_categories) > 2:
            sorted_by_count = sorted(category_mention_count.items(), key=lambda x: x[1])
            for cat, _count in sorted_by_count:
                if cat != "Awards & Certifications":
                    return cat

        return least_mentioned

    def generate_followup_question(
        self,
        final_response: str,
        user_query: str,
        current_category: str,
        conversation_history: List[Dict],
        explored_categories: List[str],
        detected_language: str = "en"
    ) -> str:
        """
        Generate a follow-up question using pre-generated questions (fast path).
        Translates to user's detected language if not English.

        Args:
            final_response: The response to the current question
            user_query: The current user query
            current_category: Category of the current question
            conversation_history: History of previous questions
            explored_categories: Categories already explored
            detected_language: Detected language code (e.g., 'en', 'de', 'fr', 'th')

        Returns:
            Generated follow-up question (translated if needed)
        """
        from prompts import FOLLOWUP_QUESTIONS_BY_CATEGORY

        recommended_category = self.get_recommended_category(
            current_category,
            explored_categories,
            conversation_history,
            user_query
        )

        # Use pre-generated questions (ALWAYS - fast path)
        # This is the only path used in production for optimal performance
        self.logger.info(f"⚡ Using pre-generated follow-up questions from category: {recommended_category}")
        question_index = len(conversation_history) % 5
        questions = FOLLOWUP_QUESTIONS_BY_CATEGORY.get(
            recommended_category,
            FOLLOWUP_QUESTIONS_BY_CATEGORY.get("General Overview", [])
        )
        if questions:
            followup_question = questions[question_index % len(questions)]
        else:
            followup_question = "What else would you like to know about Pattreeya?"

        # Translate question if user's language is not English
        if detected_language != "en":
            self.logger.info(f"🌍 Translating follow-up question to {detected_language}")
            followup_question = translate_text(followup_question, detected_language)
            self.logger.info(f"Translated follow-up: {followup_question}")

        self.logger.info(f"Follow-up selected: {followup_question}")
        return followup_question


# ============================================================================
# FOLLOW-UP GENERATION ORCHESTRATOR
# ============================================================================

def generate_followup_with_skip_check(
    state: Dict[str, Any],
    final_response: str,
    config: Optional[ConfigManager] = None,
    perf_timer: Optional[Any] = None,
    logger_instance: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Handle follow-up question generation with skip_followup flag check.

    This function encapsulates the entire follow-up generation logic including:
    - Language detection from user query
    - skip_followup flag check
    - Category classification
    - FollowUpQuestionGenerator integration
    - Error handling and logging

    Args:
        state: LLMCVAgentState containing user_query, conversation_history, etc.
        final_response: The main LLM response to generate follow-up for
        config: ConfigManager (optional, defaults to get_config())
        perf_timer: Performance timer instance (optional)
        logger_instance: Logger instance (optional, defaults to module logger)

    Returns:
        Updated state dict with:
        - followup_question: Generated follow-up question
        - detected_language: Detected language code (e.g., 'en', 'de', 'fr', 'th')
        - language_confidence: Confidence score for language detection (0-1)

    Example:
        state = generate_followup_with_skip_check(
            state=state,
            final_response=final_response,
            config=config,
            perf_timer=perf_timer,
            logger_instance=logger
        )
    """
    import streamlit as st

    # Use provided logger or module logger
    log = logger_instance or logger

    # Set default empty followup_question
    state["followup_question"] = ""

    # Check if follow-up generation is skipped
    skip_followup = st.session_state.get("skip_followup", False)

    if skip_followup:
        log.info(f"⚡ Follow-up generation skipped (skip_followup flag enabled)")
        return state

    # Follow-up generation is enabled
    if perf_timer:
        log.info(f"\n{'='*80}")
        log.info(f"⏱️  FOLLOW-UP GENERATION PHASE STARTING...")
        log.info(f"{'='*80}")
        perf_timer.start("followup_generation_total")

    try:
        # Get configuration
        if config is None:
            config = get_config()

        conversation_history = state.get("conversation_history", [])
        user_query = state.get("user_query", "")

        # ===== LANGUAGE DETECTION =====
        detected_lang, lang_confidence = detect_language(user_query)
        state["detected_language"] = detected_lang
        state["language_confidence"] = lang_confidence
        log.info(f"\n🌍 Detected language: {detected_lang} (confidence: {lang_confidence:.2f})")

        # ===== SOLUTION 1: Semantic Category Classification =====
        if perf_timer:
            log.info(f"\n⏱️  Category Classification Phase...")
            perf_timer.start("category_classification_phase")

        skip_classifier = st.session_state.get("skip_question_classification", True)

        if skip_classifier:
            current_category = "General Overview"
            log.info(f"⚡ Classifier skipped (skip_question_classification enabled)")
            log.info(f"Current question category (using default): {current_category}")
        else:
            if "question_classifier" not in st.session_state:
                st.session_state.question_classifier = QuestionCategoryClassifier(config)

            classifier = st.session_state.question_classifier
            current_category = classifier.classify_question(state['user_query'])
            log.info(f"Current question category (LLM-classified): {current_category}")

        # Extract explored categories
        explored_categories_list = []
        if not skip_classifier:
            for item in conversation_history:
                user_query = item.get("user_query", "")
                category = classifier.classify_question(user_query)
                if category not in explored_categories_list:
                    explored_categories_list.append(category)

        explored_categories_str = ", ".join(explored_categories_list) if explored_categories_list else "None yet"
        log.info(f"Explored categories (LLM-classified): {explored_categories_str}")

        # ===== INTEGRATION: FollowUpQuestionGenerator =====
        generator = FollowUpQuestionGenerator(config)

        followup_question = generator.generate_followup_question(
            final_response=final_response,
            user_query=state['user_query'],
            current_category=current_category,
            conversation_history=conversation_history,
            explored_categories=explored_categories_list,
            detected_language=detected_lang
        )

        # Store result
        state["followup_question"] = followup_question
        log.info(f"Final Follow-up: {followup_question}\n")

    except Exception as e:
        log.warning(f"Failed to generate follow-up question: {e}")
        state["followup_question"] = ""

    finally:
        # Always log timing summary
        if perf_timer:
            perf_timer.end("followup_generation_total", verbose=True)
            log.info(f"{'='*80}")
            log.info(f"⏱️  FOLLOW-UP GENERATION PHASE COMPLETED")
            log.info(f"{'='*80}\n")

    return state