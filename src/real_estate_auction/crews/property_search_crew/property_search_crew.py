import os
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from real_estate_auction.rag.rag_tool import PropertySearchTool
from real_estate_auction.types import PropertySearchResult


@CrewBase
class PropertySearchCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    llm = LLM(
        model=os.getenv("MODEL", "ollama/llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        provider="ollama",
    )


    @agent
    def property_search_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["property_search_specialist"],
            llm=self.llm,
            tools=[PropertySearchTool()],
            memory=False,
            verbose=False,
            max_iterations=1,
        )

    @task
    def property_search_task(self):
        return Task(
            config=self.tasks_config["property_search_task"],
            agent=self.property_search_specialist(),
            # output_pydantic=PropertySearchResult,
            output_file="property_search_result.json",
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )