"""
Insights generation services for micro and high-level analytics.
"""

import json
from typing import Dict, List, Any, Optional, Literal
from app.core.config import get_settings
from app.core.dependencies import get_openai_client, get_gemini_client
from app.core.exceptions import OpenAIError, ServiceError
from app.core.logging import get_logger
from app.shared.utils import retry_with_backoff, Timer

settings = get_settings()
logger = get_logger(__name__)

# Type aliases for clarity
Sentiment = Literal["Positive", "Negative", "Neutral"]


class MicroInsightEngine:
    """Service for generating micro-level client insights."""
    
    def __init__(self):
        self.client = get_openai_client()
        self.model = settings.OPENAI_MODEL_MINI
        self.temperature = 0.4
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def classify_sentiment(self, text: str) -> Sentiment:
        """
        Classify aggregate sentiment of provided text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Sentiment classification
        """
        try:
            with Timer("sentiment_classification"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Classify sentiment as Positive, Negative, or Neutral. Only return one word."
                        },
                        {"role": "user", "content": text}
                    ],
                    temperature=0
                )
                
                result = response.choices[0].message.content.strip()
                
                # Validate result
                if result not in ["Positive", "Negative", "Neutral"]:
                    logger.warning(f"Invalid sentiment classification: {result}, defaulting to Neutral")
                    return "Neutral"
                
                return result
                
        except Exception as e:
            logger.error(f"Sentiment classification failed: {e}")
            raise OpenAIError(f"Sentiment classification failed: {str(e)}")
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def generate_insight(
        self,
        client_info: Dict[str, Any],
        messages: List[Dict[str, Any]],
        previous_insight: Optional[str],
        current_sentiment: Sentiment
    ) -> str:
        """
        Generate a single-sentence, personalized micro insight.
        
        Args:
            client_info: Client profile information
            messages: Recent conversation messages
            previous_insight: Previously generated insight
            current_sentiment: Current sentiment assessment
            
        Returns:
            Single-sentence micro insight
        """
        try:
            # Prepare context data
            content = {
                "client_profile": client_info or {},
                "previous_insight": previous_insight or "",
                "recent_messages": [
                    {k: v for k, v in m.items() if k in ("timestamp", "sender", "body", "content")}
                    for m in messages[-200:]  # Limit to recent messages
                ],
                "current_sentiment": current_sentiment,
            }
            
            with Timer("insight_generation"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Micro Insight Engine. Generate exactly one sentence that a case manager can read "
                                "to instantly understand what's going on with the client right now. "
                                "Embed the client's current sentiment naturally (Positive, Neutral, or Negative) in the sentence. "
                                "Focus on tone, preferences, and the most relevant actionable cue. "
                                "Avoid repeating the previous insight verbatim; refine or extend it if useful."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                "Generate a single-sentence micro insight based on this JSON (return one sentence only, no labels):\n"
                                + json.dumps(content, ensure_ascii=False)
                            )
                        }
                    ],
                    temperature=self.temperature
                )
                
                insight = response.choices[0].message.content.strip()
                
                # Normalize to a single sentence
                if insight and insight[-1] not in ".!?":
                    insight += "."
                
                # Ensure sentiment is visible in the insight
                if current_sentiment not in insight:
                    insight = f"Sentiment: {current_sentiment} — {insight}"
                
                logger.info(
                    "Micro insight generated",
                    extra={
                        "client_id": client_info.get("client_id", "unknown"),
                        "sentiment": current_sentiment,
                        "insight_length": len(insight)
                    }
                )
                
                return insight
                
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            raise OpenAIError(f"Insight generation failed: {str(e)}")
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def adjust_sentiment(
        self,
        previous: Optional[Sentiment],
        messages: List[Dict[str, Any]]
    ) -> Sentiment:
        """
        Decide if sentiment should change based on recent messages.
        
        Args:
            previous: Previously stored sentiment
            messages: Recent conversation messages
            
        Returns:
            Updated sentiment classification
        """
        try:
            # Aggregate text for classification
            text_blob = "\n".join(
                str(m.get("body") or m.get("content", ""))
                for m in messages[-500:]  # Last 500 messages
            )
            
            if not text_blob.strip():
                return previous or "Neutral"
            
            current = await self.classify_sentiment(text_blob)
            
            if previous is None:
                return current
            
            # Ask model whether to keep or change sentiment
            with Timer("sentiment_adjustment"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You adjust a client's sentiment between Positive, Neutral, Negative. "
                                "Base it on recent interactions. If unclear, keep the previous value. "
                                "Return exactly one word: Positive, Neutral, or Negative."
                            )
                        },
                        {
                            "role": "user",
                            "content": json.dumps({"previous": previous, "observed": current})
                        }
                    ],
                    temperature=0
                )
                
                result = response.choices[0].message.content.strip()
                
                # Validate result
                if result not in ["Positive", "Negative", "Neutral"]:
                    logger.warning(f"Invalid sentiment adjustment: {result}, keeping previous")
                    return previous
                
                return result
                
        except Exception as e:
            logger.error(f"Sentiment adjustment failed: {e}")
            # Return previous sentiment on error
            return previous or "Neutral"
    
    async def run_micro_insight_engine(
        self,
        client_id: str,
        client_profile: Dict[str, Any],
        messages: List[Dict[str, Any]],
        previous_insight: Optional[str] = None,
        previous_sentiment: Optional[Sentiment] = None
    ) -> str:
        """
        Entry point for micro insight generation.
        
        Args:
            client_id: Unique client identifier
            client_profile: Client profile data
            messages: Recent conversation messages
            previous_insight: Previously generated insight
            previous_sentiment: Previously stored sentiment
            
        Returns:
            Single-sentence micro insight with embedded sentiment
        """
        try:
            with Timer("complete_micro_insight"):
                # Adjust sentiment based on recent messages
                sentiment = await self.adjust_sentiment(previous_sentiment, messages)
                
                # Generate insight
                insight = await self.generate_insight(
                    client_info={"client_id": client_id, **client_profile},
                    messages=messages,
                    previous_insight=previous_insight,
                    current_sentiment=sentiment
                )
                
                logger.info(
                    "Micro insight engine completed",
                    extra={
                        "client_id": client_id,
                        "final_sentiment": sentiment,
                        "previous_sentiment": previous_sentiment,
                        "sentiment_changed": sentiment != previous_sentiment
                    }
                )
                
                return insight
                
        except Exception as e:
            logger.error(f"Micro insight engine failed: {e}")
            # Return a safe fallback insight
            return f"Sentiment: {previous_sentiment or 'Neutral'} — Recent client interaction requires review."


class HighLevelInsightEngine:
    """Service for generating high-level firm insights and reports using Gemini 2.5 Pro."""
    
    def __init__(self):
        self.client = get_gemini_client()
        self.model = settings.GEMINI_MODEL
        self.temperature = settings.GEMINI_TEMPERATURE
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def generate_high_level_insights(
        self,
        firm_name: str,
        admin_names: List[str],
        report_period: str,
        analysis_date: str,
        firm_wide_data: Dict[str, Any],
        user_performance_data: List[Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive high-level insights for firm leadership using Gemini 2.5 Pro.
        
        Args:
            firm_name: Name of the law firm
            admin_names: List of administrator names
            report_period: Reporting period description
            analysis_date: Date of analysis
            firm_wide_data: Firm-wide analytics data
            user_performance_data: Individual user performance data
            
        Returns:
            Formatted high-level insights report
        """
        try:
            admin_names_str = ", ".join(admin_names)
            
            prompt = f"""
    You are Gemini 2.5 Pro, acting as Arviso's High-Level Insight Engine, a sophisticated AI business analyst for law firms. Your audience is busy firm leadership. Your goal is to move beyond raw data to provide interpreted, actionable business intelligence.

    **Your Task:**
    Analyze the provided JSON data blobs for {firm_name} for the period {report_period}. Your task is to identify the most statistically significant, business-critical patterns and anomalies. Do not just list the data; interpret it.
    
    1.  Write a concise **Executive Summary** that covers all major findings.
    2.  For each major finding, create a section with a compelling title. In each section, you must answer three questions:
        -   **What I'm seeing:** Describe the data pattern clearly and simply.
        -   **Why it matters:** Explain the business impact (e.g., risk to client retention, opportunity for efficiency, potential liability).
        -   **How to fix it:** Provide a clear, concrete, and actionable recommendation.
    3.  Synthesize all "How to fix it" recommendations into a prioritized list of **Summary of Action Items**.

    **Input Data for Analysis:**
    -   **Firm-Wide Data:** {json.dumps(firm_wide_data)}
    -   **User Performance & Micro-Level Insights:** {json.dumps(user_performance_data)}

    **Required Output Format:**
    Generate the report using the following structure EXACTLY. The final output will be placed in an email. The email is sent to Ai@Arviso.ai for screening, and the body must note who the final recipients are. Do not add any other commentary.

    **--- REQUIRED EMAIL BODY STRUCTURE ---**
    This monthly insight report for {firm_name} is ready for your review. If approved, please forward to the designated recipients:
    {', '.join([f'{name} <email@example.com>' for name in admin_names])}

    **Arviso High-Level Insight: {firm_name}**
    **Report for Period:** {report_period}
    **Date of Analysis:** {analysis_date}
    **Prepared For:** {admin_names_str}, {firm_name}
    
    ---
    
    **Executive Summary:**
    [Your 2-4 sentence summary of all key findings and their implications goes here.]
    
    ---
    
    **1 - [Compelling Title for Finding #1]**
    
    **What I'm seeing:** [Clear, simple description of the data pattern.]
    
    **Why it matters:** [Explanation of the business impact.]
    
    **How to fix it:** [A concrete, actionable recommendation.]
    
    ---
    
    (Continue with more numbered sections for each significant insight you discover.)
    
    ---
    
    **Summary of Action Items:**
    
    **Priority 1:** [The most urgent "How to fix it" recommendation.]
    **Priority 2:** [The second-most urgent recommendation.]
    """

            with Timer("gemini_high_level_insight_generation"):
                # Use Gemini's generate_content method
                response = await self._generate_content_async(prompt.strip())
                
                insights_report = response.text.strip()
                
                logger.info(
                    "High-level insights generated with Gemini 2.5 Pro",
                    extra={
                        "firm_name": firm_name,
                        "report_period": report_period,
                        "admin_count": len(admin_names),
                        "report_length": len(insights_report),
                        "model": self.model
                    }
                )
                
                return insights_report
                
        except Exception as e:
            logger.error(f"Gemini high-level insight generation failed: {e}")
            raise ServiceError(f"High-level insight generation failed: {str(e)}")
    
    async def _generate_content_async(self, prompt: str):
        """
        Async wrapper for Gemini's generate_content method.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            Generated content response
        """
        import asyncio
        
        # Run the synchronous Gemini call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            self.client.generate_content, 
            prompt
        )
        return response
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def generate_summary_insights(
        self,
        firm_data: Dict[str, Any],
        time_period: str = "monthly"
    ) -> Dict[str, Any]:
        """
        Generate summary insights for quick overview using Gemini.
        
        Args:
            firm_data: Aggregated firm data
            time_period: Time period for analysis
            
        Returns:
            Dictionary with summary insights
        """
        try:
            prompt = f"""
            Analyze the provided firm data and generate 3-5 key summary insights. 
            Focus on the most important trends, risks, and opportunities. 
            Return a JSON object with insights as an array of objects containing 'title' and 'description'.
            
            Analyze this {time_period} firm data:
            {json.dumps(firm_data)}
            
            Format your response as valid JSON only, no additional text.
            """
            
            with Timer("gemini_summary_insight_generation"):
                response = await self._generate_content_async(prompt)
                
                # Parse JSON response
                try:
                    result = json.loads(response.text.strip())
                except json.JSONDecodeError:
                    # If direct JSON parsing fails, try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                    else:
                        raise json.JSONDecodeError("No valid JSON found in response")
                
                logger.info(
                    "Summary insights generated with Gemini",
                    extra={
                        "time_period": time_period,
                        "insights_count": len(result.get("insights", [])),
                        "model": self.model
                    }
                )
                
                return result
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini summary insights response: {e}")
            raise ServiceError("Invalid response format from summary insights")
        except Exception as e:
            logger.error(f"Gemini summary insight generation failed: {e}")
            raise ServiceError(f"Summary insight generation failed: {str(e)}")
