from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PropertyFeatures(BaseModel):
    sqm: int
    location: str
    has_terrace: bool
    near_metro: bool
    features: List[str]

class PropertyDetails(BaseModel):
    id: str
    title: str
    starting_price: int
    description: str
    specs: PropertyFeatures


class PropertyRecommendation(BaseModel):
    property_id: str
    title: str
    district: str
    property_type: str
    price_eur: int
    size_sqm: int
    bedrooms: int
    why_it_matches: str
    tradeoffs: str


class PropertySearchResult(BaseModel):
    summary: str
    recommendations: List[PropertyRecommendation]


class BidAttempt(BaseModel):
    agent_name: str
    action: Literal["bid", "pass"]
    bid_amount: Optional[int] = None
    reason: str
    confidence: float


class AuctionRound(BaseModel):
    round_number: int = 0
    highest_bid: int
    current_leader: str
    is_auction_over: bool
    status_message: str
    bid_attempts: List[BidAttempt] = Field(default_factory=list)


class AuctionSummary(BaseModel):
    property_id: str
    winner: str
    final_price: int
    starting_price: int
    history: List[BidAttempt]
    margin_over_starting: int
    total_rounds: int