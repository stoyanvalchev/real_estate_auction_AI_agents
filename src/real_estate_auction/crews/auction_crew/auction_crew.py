import os
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, task, crew
from real_estate_auction.types import AuctionRound, BidAttempt


@CrewBase
class AuctionDecisionCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    llm = LLM(
        model=os.getenv("MODEL", "ollama/llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        provider="ollama",
    )

    @agent
    def family_buyer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["family_buyer_agent"],
            llm=self.llm,
            memory=False,
            verbose=False,
        )

    @agent
    def investor_buyer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["investor_buyer_agent"],
            llm=self.llm,
            memory=False,
            verbose=False,
        )
    
    @agent
    def auction_orchestrator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["auction_orchestrator_agent"],
            llm=self.llm,
            memory=False,
            verbose=False,
        )

    @task
    def family_bid_task(self):
        return Task(
            config=self.tasks_config["family_bid_task"],
            # output_pydantic=BidAttempt
        )

    @task
    def investor_bid_task(self) -> Task:
        return Task(
            config=self.tasks_config["investor_bid_task"],
            context=[self.family_bid_task()],
            # output_pydantic=BidAttempt
        )

    @task
    def auction_orchestrator_task(self) -> Task:
        return Task(
            config=self.tasks_config["auction_orchestrator_task"],
            context=[self.family_bid_task(), self.investor_bid_task()],
            # output_pydantic=AuctionRound
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[
                self.family_bid_task(),
                self.investor_bid_task(),
                self.auction_orchestrator_task(),
            ],
            process=Process.sequential, 
            verbose=False
        )

# def get_family_bid(inputs: dict) -> BidDecision:
#     crew = AuctionDecisionCrew()
#     result = Crew(
#         agents=[crew.family_buyer_agent()],
#         tasks=[crew.family_bid_task()],
#         process=Process.sequential,
#         verbose=True,
#     ).kickoff(inputs=inputs)

#     if getattr(result, "pydantic", None) is not None:
#         return result.pydantic
#     raise ValueError("Family bid did not return structured output.")


# def get_investor_bid(inputs: dict) -> BidDecision:
#     crew = AuctionDecisionCrew()
#     result = Crew(
#         agents=[crew.investor_buyer_agent()],
#         tasks=[crew.investor_bid_task()],
#         process=Process.sequential,
#         verbose=True,
#     ).kickoff(inputs=inputs)

#     if getattr(result, "pydantic", None) is not None:
#         return result.pydantic
#     raise ValueError("Investor bid did not return structured output.")