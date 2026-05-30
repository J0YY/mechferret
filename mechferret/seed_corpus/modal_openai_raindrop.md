# Sponsor-Aligned Implementation Notes

Modal is useful when the research loop needs parallel ingestion, model calls, or
compute-heavy parsing across a large corpus. A hackathon system can start with a
local runner and include a Modal function for scale-out research jobs.

OpenAI Responses API is a strong fit for live web search and agentic tool use.
A robust project should still work without an API key, then upgrade to live
search when credentials are present.

Raindrop Workshop is valuable for showing how the agent behaved during a run.
Tracing the planner, retriever, extractor, critic, and synthesizer phases turns
agent behavior into an inspectable timeline rather than terminal noise.

