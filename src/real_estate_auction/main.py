import re
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, ValidationError
from crewai.flow.flow import Flow, listen, router, start
import json
from real_estate_auction.crews.auction_crew.auction_crew import AuctionDecisionCrew
from real_estate_auction.crews.property_search_crew.property_search_crew import PropertySearchCrew
from real_estate_auction.types import PropertySearchResult, PropertyRecommendation, AuctionRound, AuctionSummary

import joblib
import pandas as pd
import json



class AuctionState(BaseModel):
    user_query: str = ""
    selected_intent: str = ""
    search_result: Optional[dict] = None

    current_highest_bid: int = 0
    highest_bidder: str = "No Bids Yet"
    property_details: str = ""
    round_history: List[AuctionRound] = []
    is_auction_over: bool = False
    status_update: str = ""

    selected_property_id: str = ""
    starting_price: int = 0
    min_increment: int = 1000
    max_rounds: int = 3
    predicted_market_value: int = 0

class PropertyAuctionFlow(Flow[AuctionState]):

    def predict_property_value(self):
        
        if not hasattr(self, "rf_model"):
            self.rf_model = joblib.load("rf_property_model.pkl")
            self.encoder = joblib.load("rf_encoder.pkl")

        prop =json.loads(self.state.property_details)
        df = pd.DataFrame([prop])

        drop_cols = [
            "property_id","title","description",
            "inspection_notes","legal_notes",
            "features","nearby_transport","city", "price_eur"
        ]

        categorical_cols = [
            "district","property_type","condition",
            "furnished","heating","terrace",
            "elevator","parking"
        ]

        df = df.drop(columns=drop_cols)

        encoded = self.encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(categorical_cols)
        )

        X = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
        predicted_price = int(self.rf_model.predict(X)[0])

        return predicted_price



    @start()
    def select_mode(self):
        print("\n--- Welcome to the Real Estate AI Portal ---")
        intent_map = {
            "search": "search",
            "s": "search",
            "auction": "auction",
            "a": "auction",
        }

        with open("data/properties.json", "r", encoding="utf-8") as f:
            self.properties = json.load(f)

            self.property_index = {
            p["property_id"].lower(): p for p in self.properties
            }  

        while True:
            choice = input(
                "Would you like to 'search' for properties or start an 'auction'? "
            ).strip().lower()

            intent = intent_map.get(choice)

            if intent:
                self.state.selected_intent = intent
                break

            print("Invalid choice. Please enter 'search' or 'auction'.")

    @router(select_mode)
    def mode_router(self):
        if self.state.selected_intent == "search":
            return "search_mode"
        return "auction_mode"

    # ---------------- SEARCH BRANCH ----------------

    @listen("search_mode")
    def collect_search_criteria(self): 
        print("\n[Mode: Search]")
        while True:
            query = input(
                "Describe the property you are looking for "
                "(e.g. 2BR in Sofia, terrace, near metro): "
            ).strip()

            if query:
                self.state.user_query = query
                break

            print("Empty query received. Please try again.")

            self.state.user_query = query

    @listen(collect_search_criteria)
    async def execute_search(self):
        result = await (
            PropertySearchCrew()
            .crew()
            .kickoff_async(
                inputs={
                    "user_query": self.state.user_query,
                    "conversation_context": "",
                }
            )
        )

        raw = re.sub(r"```(?:json)?\s*|\s*```", "", result.raw).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]

        try:
            parsed = PropertySearchResult.model_validate_json(raw)

            parsed.recommendations = [
                PropertyRecommendation(
                    property_id=prop["property_id"],
                    title=prop["title"],
                    district=prop["district"],
                    property_type=prop["property_type"],
                    price_eur=prop["price_eur"],
                    size_sqm=prop["size_sqm"],
                    bedrooms=prop["bedrooms"],
                    why_it_matches=rec.why_it_matches,
                    tradeoffs=rec.tradeoffs,
                )
                for rec in parsed.recommendations
                if (prop := self.property_index.get(rec.property_id.lower()))
            ]

            # Deduplicate recommendations by property_id
            seen_ids = set()
            unique_recs = []
            for rec in parsed.recommendations:
                if rec.property_id.lower() not in seen_ids:
                    seen_ids.add(rec.property_id.lower())
                    unique_recs.append(rec)
            parsed.recommendations = unique_recs

            # Save result to file
            output_dir = Path("data/search_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / "search_result.json"

            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "query": self.state.user_query,
                        "result": json.loads(parsed.model_dump_json()),
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"\n[Search] Result saved to {filepath}")

            # Store in state
            self.state.search_result = parsed.model_dump_json()

            # Print final output
            print(f"\nSummary: {parsed.summary}")
            for rec in parsed.recommendations:
                print(
                    f"\n  • {rec.title} | {rec.district} | "
                    f"{rec.price_eur:,} EUR | {rec.size_sqm} sqm | {rec.bedrooms} bd"
                )
                print(f"    Why: {rec.why_it_matches}")
                print(f"    Tradeoffs: {rec.tradeoffs}")

            return parsed

        except ValidationError as e:
            print("Validation error while parsing crew output:")
            print(e)
            print("Raw crew output:")
            print(raw)

            fallback = PropertySearchResult(
                summary="I found some results, but I could not format them reliably.",
                recommendations=[],
            )

            output_dir = Path("data/search_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / "search_result.json"

            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "query": self.state.user_query,
                        "result": json.loads(fallback.model_dump_json()),
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

            self.state.search_result = fallback.model_dump_json()
            print(f"\n[Search] Result saved to {filepath}")
            print(f"\nSummary: {fallback.summary}")

            return fallback

        except Exception as e:
            print("Unexpected error in execute_search:")
            print(e)
            print("Raw crew output:")
            print(raw)

            fallback = PropertySearchResult(
                summary="Something went wrong while processing the property search.",
                recommendations=[],
            )

            output_dir = Path("data/search_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / "search_result.json"

            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "query": self.state.user_query,
                        "result": json.loads(fallback.model_dump_json()),
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

            self.state.search_result = fallback.model_dump_json()
            print(f"\n[Search] Result saved to {filepath}")
            print(f"\nSummary: {fallback.summary}")

            return fallback
        
    # ---------------- AUCTION BRANCH ----------------

    @listen("auction_mode")
    def start_auction_branch(self):
        print("\n[Mode: Auction] Loading auction parameters...")

        try:
            with open(
                "src/real_estate_auction/crews/auction_crew/config/auction.yaml",
                "r",
                encoding="utf-8",
            ) as file:
                config = yaml.safe_load(file)

            auction_cfg = config.get("auction", {})
            property_ids = auction_cfg.get("property_to_auction", [])
            min_increment = auction_cfg.get("min_increment", 1000)
            max_rounds = auction_cfg.get("max_rounds", 3)

            if not property_ids:
                raise ValueError("No property_to_auction defined in config.")

            property_id = property_ids[0].lower()
            prop = self.property_index.get(property_id)

            if not prop:
                raise ValueError(
                    f"Property '{property_id}' not found in properties.json"
                )

            starting_price = prop["price_eur"] - 10000

            self.state.selected_property_id = prop["property_id"]
            self.state.property_details = json.dumps(prop, ensure_ascii=False, indent=2)
            self.state.starting_price = starting_price
            self.state.current_highest_bid = starting_price
            self.state.highest_bidder = "No Bids Yet"
            self.state.is_auction_over = False
            self.state.round_history = []
            self.state.status_update = ""
            self.state.min_increment = min_increment
            self.state.max_rounds = max_rounds
            self.state.predicted_market_value = self.predict_property_value()

            print(f"Predicted Market Value: {self.state.predicted_market_value} EUR")
            print(f"Auction live for: {prop['title']} ({prop['district']})")
            print(f"Starting Price: {self.state.starting_price} EUR")
            print(f"Minimum Increment: {self.state.min_increment} EUR")
            print(f"Max Rounds: {self.state.max_rounds}")

        except Exception as e:
            print(f"Error loading auction config: {e}")
            self.state.is_auction_over = True

    @listen(start_auction_branch)
    async def run_bidding_round(self):
        while not self.state.is_auction_over:
            round_number = len(self.state.round_history) + 1

            if round_number > self.state.max_rounds:
                print("Max rounds reached.")
                self.state.is_auction_over = True
                break

            print(f"\n--- Round {round_number} ---")

            inputs_payload = {
                "property_id": self.state.selected_property_id,
                "property_details": self.state.property_details,
                "starting_price": self.state.starting_price,
                "current_highest_bid": self.state.current_highest_bid,
                "previous_leader": self.state.highest_bidder,
                "min_increment": self.state.min_increment,
                "max_rounds": self.state.max_rounds,
                "round_number": round_number,
                "predicted_market_value": self.state.predicted_market_value,
            }

            try:
                result = await (
                    AuctionDecisionCrew()
                    .crew()
                    .kickoff_async(inputs=inputs_payload)
                )
            except Exception as e:
                print(f"Error during bidding round: {e}")
                self.state.is_auction_over = True
                break

            raw_output = getattr(result, "raw", None)
            pydantic_output = getattr(result, "pydantic", None)

            print("\n[DEBUG] Crew raw output:")
            print(raw_output)
            print("\n[DEBUG] Crew pydantic output:")
            print(pydantic_output)

            round_data = None

            if pydantic_output and isinstance(pydantic_output, AuctionRound):
                round_data = pydantic_output
            else:
                try:
                    if raw_output:
                        raw_clean = re.sub(
                            r"```(?:json)?\s*|\s*```", "", raw_output
                        ).strip()
                        start_idx, end_idx = raw_clean.find("{"), raw_clean.rfind("}")
                        if start_idx != -1 and end_idx != -1:
                            raw_clean = raw_clean[start_idx : end_idx + 1]

                        raw_dict = json.loads(raw_clean)

                        if raw_dict.get("status_message") in (None, ""):
                            raw_dict["status_message"] = f"Round {round_number} processed."

                        confidence_map = {
                            "low": 0.3,
                            "medium": 0.5,
                            "moderate": 0.6,
                            "high": 0.9,
                        }

                        for attempt in raw_dict.get("bid_attempts", []):
                            action = attempt.get("action")
                            if isinstance(action, str):
                                attempt["action"] = action.strip().lower()

                            conf = attempt.get("confidence")
                            if isinstance(conf, str):
                                attempt["confidence"] = confidence_map.get(conf.strip().lower(), 0.5)

                        # Normalize nullable / missing fields before validation
                        if raw_dict.get("current_leader") in (None, ""):
                            bid_attempts = raw_dict.get("bid_attempts", [])
                            valid_bid_attempts = [
                                a for a in bid_attempts
                                if a.get("action") == "bid" and isinstance(a.get("bid_amount"), int)
                            ]

                            if valid_bid_attempts:
                                winning_attempt = max(valid_bid_attempts, key=lambda x: x["bid_amount"])
                                raw_dict["current_leader"] = winning_attempt["agent_name"]
                            else:
                                raw_dict["current_leader"] = self.state.highest_bidder

                        if raw_dict.get("highest_bid") is None:
                            raw_dict["highest_bid"] = self.state.current_highest_bid

                        if raw_dict.get("is_auction_over") is None:
                            raw_dict["is_auction_over"] = False

                        round_data = AuctionRound.model_validate(raw_dict)

                except Exception as e:
                    print("Error: Failed to parse raw orchestrator output as AuctionRound.")
                    print(e)
                    self.state.is_auction_over = True
                    return

            if round_data is None:
                print("Error: Orchestrator did not return usable AuctionRound data.")
                self.state.is_auction_over = True
                return

            # --- Check for passes first ---
            any_pass = any(
                attempt.action == "pass" for attempt in round_data.bid_attempts
            )

            # --- Validate bids with rolling increment ---
            valid_bids = []
            round_floor = self.state.current_highest_bid

            for attempt in sorted(
                round_data.bid_attempts, key=lambda x: x.bid_amount or 0
            ):
                if attempt.action == "pass":
                    continue

                if attempt.action == "bid":
                    if attempt.bid_amount is None or not isinstance(attempt.bid_amount, int):
                        print(
                            f"[DEBUG] Invalid bid from {attempt.agent_name}: bad bid_amount."
                        )
                        continue

                    min_valid_bid = round_floor + self.state.min_increment
                    if attempt.bid_amount < min_valid_bid:
                        print(
                            f"[DEBUG] Invalid bid from {attempt.agent_name}: "
                            f"{attempt.bid_amount} < required {min_valid_bid}."
                        )
                        continue

                    round_floor = attempt.bid_amount
                    valid_bids.append(attempt)

            # --- Override orchestrator result ---
            if any_pass and not valid_bids:
                round_data.is_auction_over = True
                round_data.highest_bid = self.state.current_highest_bid
                round_data.current_leader = self.state.highest_bidder
                round_data.status_message = (
                    f"Round {round_number} complete: an agent passed with no valid bids."
                )

            elif any_pass and valid_bids:
                round_data.is_auction_over = True
                winning_bid = max(valid_bids, key=lambda x: x.bid_amount)
                round_data.highest_bid = winning_bid.bid_amount
                round_data.current_leader = winning_bid.agent_name
                round_data.status_message = (
                    f"Round {round_number} complete: {winning_bid.agent_name} wins at "
                    f"{winning_bid.bid_amount} EUR (opponent passed)."
                )

            elif valid_bids:
                winning_bid = max(valid_bids, key=lambda x: x.bid_amount)
                round_data.is_auction_over = False
                round_data.highest_bid = winning_bid.bid_amount
                round_data.current_leader = winning_bid.agent_name
                round_data.status_message = (
                    f"Round {round_number} complete: {winning_bid.agent_name} leads "
                    f"with {winning_bid.bid_amount} EUR."
                )

            else:
                round_data.is_auction_over = True
                round_data.highest_bid = self.state.current_highest_bid
                round_data.current_leader = self.state.highest_bidder
                round_data.status_message = (
                    f"Round {round_number} complete: no valid new bids."
                )

            # --- Persist state ---
            self.state.round_history.append(round_data)
            self.state.current_highest_bid = round_data.highest_bid
            self.state.highest_bidder = round_data.current_leader
            self.state.is_auction_over = round_data.is_auction_over
            self.state.status_update = round_data.status_message

            print(f"Update: {round_data.status_message}")
            print(
                f"Leading: {self.state.highest_bidder} "
                f"at {self.state.current_highest_bid} EUR"
            )

            if round_data.bid_attempts:
                print("[DEBUG] Bid attempts received:")
                for attempt in round_data.bid_attempts:
                    if attempt.action == "bid":
                        print(
                            f" - {attempt.agent_name}: BID {attempt.bid_amount} "
                            f"(confidence={attempt.confidence})"
                        )
                    else:
                        print(
                            f" - {attempt.agent_name}: PASS "
                            f"(confidence={attempt.confidence})"
                        )

    @listen(run_bidding_round)
    def announce_winner(self):
        all_attempts = []
        for round_data in self.state.round_history:
            all_attempts.extend(round_data.bid_attempts)

        summary = AuctionSummary(
            property_id=self.state.selected_property_id,
            winner=self.state.highest_bidder,
            final_price=self.state.current_highest_bid,
            starting_price=self.state.starting_price,
            history=all_attempts,
            margin_over_starting=(
                self.state.current_highest_bid - self.state.starting_price
            ),
            total_rounds=len(self.state.round_history),
        )

        output_dir = Path("data/auction_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "auction_result.json"

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(
                json.loads(summary.model_dump_json()),
                fh,
                indent=2,
                ensure_ascii=False,
            )

        print("\n" + "=" * 40)
        print(" AUCTION CLOSED")
        print("=" * 40)
        print(f"PROPERTY : {summary.property_id}")
        print(f"WINNER : {summary.winner}")
        print(f"FINAL PRICE : {summary.final_price} EUR")
        print(f"START PRICE : {summary.starting_price} EUR")
        print(f"MARGIN : {summary.margin_over_starting} EUR")
        print(f"ROUNDS : {summary.total_rounds}")
        print("=" * 40)
        print(f"[Auction] Result saved to {filepath}")


def kickoff():
    flow = PropertyAuctionFlow()
    flow.kickoff()


if __name__ == "__main__":
    kickoff()
    