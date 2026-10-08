"""Streamlit app simulating a digital-twin service-innovation workshop with a judge and CLV check."""
from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
import zipfile
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DEFAULT_USERS_FILE = APP_DIR / "users.xlsx"
MODEL_NAME = "gemini-3.1-flash-lite"
CAC_PER_CUSTOMER = 100.0
PANEL_SIZE = 10
DISCUSSION_SECONDS = 15
YEARS = range(1, 6)
INNOVATION_TYPES = {
    "Market Penetration / Market Share Building": {
        "offering": "Existing services", "market": "Existing customers",
        "action": "Sell more of the existing services in existing markets.",
    },
    "Market Development": {
        "offering": "Existing services", "market": "New customers",
        "action": "Find or create new markets and sell the existing services in these new markets.",
    },
    "Service Development": {
        "offering": "New services", "market": "Existing customers",
        "action": "Add new services and sell them in existing markets.",
    },
    "Diversification": {
        "offering": "New services", "market": "New customers",
        "action": "Add new services and sell them in new markets.",
    },
}
DOWNSTREAM_KEYS = ("discussion", "collated", "idea_box", "evaluation", "clv_innovation", "final_decision")


class DiscussionTurn(BaseModel):
    message: str = Field(description="Two to three natural first-person sentences spoken in the discussion")


class CollatedIdeas(BaseModel):
    summary: str = Field(description="Two to three sentences summarizing where the discussion landed")
    ideas: list[str] = Field(
        min_length=3, max_length=10,
        description="Distinct, self-contained idea points from the discussion, including refinements and concerns raised",
    )


class CitedPoint(BaseModel):
    point: str = Field(description="The feedback point itself, without any citation in the text")
    citation: str = Field(
        description='The exact bracketed slide label that supports this point, e.g. "Chapter 3, Slide 9"; '
        'an empty string if no slide genuinely supports it'
    )
    evidence: str = Field(
        description="A phrase of 4-15 words copied word for word from the cited slide; an empty string if no citation"
    )


class Evaluation(BaseModel):
    overall_thoughts: str = Field(description="The judge's overall view of the idea in three to five sentences")
    slide_concepts_applied: list[CitedPoint] = Field(description="Course concepts from the slides used in the evaluation")
    positives: list[CitedPoint] = Field(description="Strengths of the idea")
    negatives: list[CitedPoint] = Field(description="Weaknesses and risks of the idea")
    suggested_changes: list[CitedPoint] = Field(description="Concrete changes that would improve the idea")
    decision: Literal["GO", "NO GO"]
    requests_business_analysis: bool = Field(
        description="True only when the decision is GO and a customer lifetime value analysis is needed before execution"
    )
    analysis_request: str = Field(
        description="If requesting analysis, what the judge wants the CLV analysis to show; otherwise an empty string"
    )


class FinalDecision(BaseModel):
    decision: Literal["GO", "NO GO"]
    rationale: str = Field(description="Three to five sentences explaining how the CLV report shaped the decision")
    conditions: list[CitedPoint] = Field(description="Conditions, safeguards, or next steps attached to the decision")


class RetentionResponse(BaseModel):
    year_1_probability: int = Field(ge=0, le=100, description="Calibrated probability of returning in year 1")
    year_2_probability: int = Field(ge=0, le=100, description="Conditional probability of returning in year 2")
    year_3_probability: int = Field(ge=0, le=100, description="Conditional probability of returning in year 3")
    year_4_probability: int = Field(ge=0, le=100, description="Conditional probability of returning in year 4")
    year_5_probability: int = Field(ge=0, le=100, description="Conditional probability of returning in year 5")
    reason: str = Field(description="One concise sentence explaining the decisions")


def clean_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


@st.cache_data(show_spinner=False)
def load_default_profiles() -> pd.DataFrame:
    return pd.read_excel(DEFAULT_USERS_FILE)


def deck_title(path: Path) -> str:
    chapter = re.search(r"(?:Chapter|Ch)\s*(\d+)", path.stem, flags=re.I)
    part = re.search(r"Part\s+([IVX]+)", path.stem)
    title = f"Chapter {chapter.group(1)}" if chapter else path.stem
    return f"{title} Part {part.group(1)}" if part else title


def deck_slide_texts(archive: zipfile.ZipFile) -> list[str]:
    """Return each slide's text in the order PowerPoint shows them (presentation.xml), not file-name order."""
    presentation = archive.read("ppt/presentation.xml").decode("utf-8", errors="ignore")
    relationships = archive.read("ppt/_rels/presentation.xml.rels").decode("utf-8", errors="ignore")
    targets = {}
    for tag in re.findall(r"<Relationship\b[^>]*>", relationships):
        rid, target = re.search(r'\bId="([^"]+)"', tag), re.search(r'\bTarget="([^"]+)"', tag)
        if rid and target:
            path = target.group(1).lstrip("/")
            targets[rid.group(1)] = path if path.startswith("ppt/") else f"ppt/{path}"
    texts = []
    for rid in re.findall(r'<p:sldId\b[^>]*\br:id="([^"]+)"', presentation):
        xml = archive.read(targets[rid]).decode("utf-8", errors="ignore")
        paragraphs = [
            " ".join(unescape(run) for run in re.findall(r"<a:t>([^<]*)</a:t>", paragraph))
            for paragraph in re.findall(r"<a:p>.*?</a:p>", xml, flags=re.S)
        ]
        texts.append(" | ".join(line.strip() for line in paragraphs if line.strip()))
    return texts


def pptx_files(folder: Path) -> list[Path]:
    try:
        return sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.casefold() == ".pptx" and not path.name.startswith("~$")
        )
    except OSError:
        return []


def find_slide_files() -> list[Path]:
    """Use a slides/ folder (any letter case) next to or above the app; otherwise decks sitting beside the app."""
    for base in (APP_DIR, APP_DIR.parent):
        try:
            folders = [path for path in base.iterdir() if path.is_dir() and path.name.casefold() == "slides"]
        except OSError:
            continue
        for folder in folders:
            if files := pptx_files(folder):
                return files
    return pptx_files(APP_DIR)


def slide_file_listing() -> str:
    """Describe what is next to the app, to explain a missing-slides error."""
    try:
        entries = sorted(path.name + ("/" if path.is_dir() else "") for path in APP_DIR.iterdir())
    except OSError as exc:
        return str(exc)
    return ", ".join(entries) or "(empty)"


def load_slides() -> tuple[list[dict], list[str]]:
    # File names, sizes, and times form the cache key, so uploading new decks refreshes the slides.
    files = find_slide_files()
    return read_slides(tuple((str(path), path.stat().st_mtime, path.stat().st_size) for path in files))


@st.cache_data(show_spinner=False)
def read_slides(files: tuple[tuple[str, float, int], ...]) -> tuple[list[dict], list[str]]:
    """Return every slide as {label, text}, plus any decks that could not be read."""
    decks, skipped = [], []
    for path in (Path(name) for name, _, _ in files):
        try:
            with zipfile.ZipFile(path) as archive:
                decks.append((path, deck_slide_texts(archive)))
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            skipped.append(f"{path.name} ({exc})")
    titles = [deck_title(path) for path, _ in decks]
    slides = []
    for (path, texts), title in zip(decks, titles):
        # Several decks for one chapter get the file name so each label stays unique.
        if titles.count(title) > 1:
            title = f"{title} [{path.stem}]"
        for number, text in enumerate(texts, start=1):
            if text:
                slides.append({"label": f"{title}, Slide {number}", "text": text})
    return slides, skipped


def slides_for_prompt(slides: list[dict]) -> str:
    chapters = sorted({slide["label"].rsplit(", Slide", 1)[0] for slide in slides})
    lines = "\n".join(f"[{slide['label']}] {slide['text']}" for slide in slides)
    return f"Decks available (no other chapters exist): {'; '.join(chapters)}\n\n{lines}"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def verify_citation(item: dict, slides: list[dict]) -> dict:
    """Keep a citation only if its evidence phrase really appears on a slide; re-point it if on another slide."""
    evidence = normalize(item.get("evidence", ""))
    claimed = next((slide for slide in slides if normalize(slide["label"]) == normalize(item.get("citation", ""))), None)
    if len(evidence.split()) >= 3:
        if claimed and evidence in normalize(claimed["text"]):
            return {**item, "citation": claimed["label"], "status": "verified"}
        match = next((slide for slide in slides if evidence in normalize(slide["text"])), None)
        if match:
            return {**item, "citation": match["label"], "status": "corrected"}
    return {**item, "citation": "", "status": "removed" if item.get("citation") else "none"}


def verify_points(record: dict, fields: tuple[str, ...], slides: list[dict]) -> dict:
    return {**record, **{field: [verify_citation(item, slides) for item in record[field]] for field in fields}}


def matching_column(dataframe: pd.DataFrame, wanted: str) -> str | None:
    names = {str(column).strip().casefold(): str(column) for column in dataframe.columns}
    return names.get(wanted.casefold())


def age_group_for(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    age = pd.to_numeric(value, errors="coerce")
    if pd.isna(age):
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if not numbers:
            return text or None
        age = float(numbers[0])
    if age < 18:
        return "Under 18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


def filter_profiles(dataframe: pd.DataFrame) -> pd.DataFrame:
    fields = (
        ("Age group", matching_column(dataframe, "Age"), age_group_for),
        ("Location", matching_column(dataframe, "Location"), clean_value),
        ("Gender", matching_column(dataframe, "Gender"), clean_value),
    )
    mask = pd.Series(True, index=dataframe.index)
    for container, (label, source, transform) in zip(st.columns(3), fields):
        if not source:
            container.caption(f"{label} filter unavailable.")
            continue
        values = dataframe[source].map(transform)
        options = sorted(set(values.dropna()), key=lambda item: str(item).casefold())
        selected = container.multiselect(label, options, placeholder=f"All {label.lower()}s", key=f"clv_filter_{label}")
        if selected:
            mask &= values.isin(selected)

    matches = dataframe.loc[mask].copy()
    if matches.empty:
        st.warning("No profiles match the selected filters.")
        return matches
    count = int(st.number_input(
        "Number of profiles to simulate", 1, len(matches), min(len(matches), 100), 1,
        help="A reproducible random sample is used when fewer profiles are selected.",
    ))
    if count < len(matches):
        matches = matches.sample(count, random_state=42).sort_index()
    return matches.reset_index(drop=True)


def configured_api_key() -> str:
    try:
        return str(st.secrets["GOOGLE_API_KEY"])
    except (KeyError, FileNotFoundError):
        return ""


def make_llm(api_key: str, schema: type[BaseModel], temperature: float, timeout: float | None = None):
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME, temperature=temperature, google_api_key=api_key,
        timeout=timeout, max_retries=1 if timeout else 2,
    ).with_structured_output(schema)


def invoke_with_retry(llm, prompt: str, attempts: int = 4):
    for attempt in range(attempts):
        try:
            return llm.invoke(prompt)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("The model did not return a response.")


def select_panel(dataframe: pd.DataFrame) -> list[dict]:
    rows = dataframe.sample(min(PANEL_SIZE, len(dataframe)), random_state=random.randint(0, 10**9))
    panel = []
    for number, (_, row) in enumerate(rows.iterrows(), start=1):
        profile = {str(column): value for column, raw in row.items() if (value := clean_value(raw)) is not None}
        panel.append({
            "name": f"Twin {number}",
            "profile": profile,
            "label": f"Twin {number} — " + ", ".join(profile.values()),
        })
    return panel


def profile_lines(profile: dict) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in profile.items()) or "- No profile fields"


def firm_context(context: dict) -> str:
    details = INNOVATION_TYPES[context["innovation_type"]]
    return f"""Firm: {context['firm']}
Firm description: {context['description']}
Firm mission: {context['mission']}
Selected service-innovation strategy: {context['innovation_type']}
({details['offering']} for {details['market']}: {details['action']})"""


def discussion_prompt(context: dict, speaker: dict, transcript: list[dict], move: str, target: str | None) -> str:
    history = "\n".join(f"{turn['speaker']}: {turn['message']}" for turn in transcript) or "(No one has spoken yet.)"
    instructions = {
        "opens": "You speak first. Propose one concrete, specific service-innovation idea that fits the selected strategy.",
        "builds on": f"Build on {target}'s point: extend it, combine it with another point, or make it more concrete. Address {target} by name.",
        "counters": (
            f"Respectfully counter {target}'s point: name a weakness, risk, or customer concern from your perspective, "
            f"then propose an alternative or a fix. Address {target} by name."
        ),
    }
    return f"""You are {speaker['name']}, a digital twin of a real customer in a fast brainstorming session on service innovation.

Your profile:
{profile_lines(speaker['profile'])}

{firm_context(context)}

Discussion so far:
{history}

Your move: {instructions[move]}
Speak in the first person as this customer, in 2-3 sentences, grounded in your profile and the firm's mission.
Stay within the selected strategy. Do not repeat ideas already stated. Do not mention being an AI or a simulation."""


def run_discussion(context: dict, panel: list[dict], api_key: str) -> list[dict]:
    """Twins take turns for DISCUSSION_SECONDS; each turn either builds on or counters an earlier speaker."""
    llm = make_llm(api_key, DiscussionTurn, temperature=0.9, timeout=12)
    names = [member["name"] for member in panel]
    members = {member["name"]: member for member in panel}
    unheard = names.copy()
    transcript: list[dict] = []
    progress = st.progress(0.0, text=f"Discussion running: {DISCUSSION_SECONDS} seconds remaining")
    start = time.monotonic()
    previous = None
    while time.monotonic() - start < DISCUSSION_SECONDS:
        # Prefer twins who have not spoken yet so the whole panel jumps in.
        candidates = [name for name in unheard if name != previous] or [name for name in names if name != previous]
        speaker = random.choice(candidates)
        if transcript:
            move = random.choices(["builds on", "counters"], weights=[0.6, 0.4])[0]
            recent = [turn["speaker"] for turn in transcript[-3:] if turn["speaker"] != speaker]
            target = random.choice(recent or [turn["speaker"] for turn in transcript if turn["speaker"] != speaker] or [previous])
        else:
            move, target = "opens", None
        try:
            response = llm.invoke(discussion_prompt(context, members[speaker], transcript, move, target))
        except Exception as exc:
            st.warning(f"{speaker} could not respond: {exc}")
            time.sleep(1)
            continue
        turn = {"speaker": speaker, "move": move, "target": target, "message": response.message.strip()}
        transcript.append(turn)
        render_turn(turn)
        if speaker in unheard:
            unheard.remove(speaker)
        previous = speaker
        elapsed = time.monotonic() - start
        progress.progress(
            min(elapsed / DISCUSSION_SECONDS, 1.0),
            text=f"Discussion running: {max(0, DISCUSSION_SECONDS - elapsed):.0f} seconds remaining",
        )
    progress.empty()
    return transcript


def render_turn(turn: dict) -> None:
    with st.chat_message("user", avatar="🧑"):
        action = "opens the discussion" if turn["move"] == "opens" else f"{turn['move']} {turn['target']}"
        st.markdown(f"**{turn['speaker']}** · _{action}_")
        st.write(turn["message"])


def collate_ideas(context: dict, judge: dict, transcript: list[dict], api_key: str) -> CollatedIdeas:
    history = "\n".join(f"{turn['speaker']} ({turn['move']} {turn['target'] or ''}): {turn['message']}" for turn in transcript)
    prompt = f"""You are {judge['name']}, a digital-twin customer who also acts as the judge of this brainstorming session.

Your profile:
{profile_lines(judge['profile'])}

{firm_context(context)}

Full discussion transcript:
{history}

Collate the discussion. Merge overlapping suggestions, keep each distinct idea as a self-contained point
(what the service is, who it is for, and how it works), and fold in the refinements and counterpoints that were raised.
Do not add ideas no one discussed."""
    return invoke_with_retry(make_llm(api_key, CollatedIdeas, temperature=0.3), prompt)


CITATION_RULES = """Citation rules (they are checked automatically against the slide text):
- Give every feedback point a citation copied exactly from a bracketed slide label above, e.g. "Chapter 3, Slide 9".
- In evidence, copy 4-15 consecutive words word for word from that same slide, so the team can find it.
- Never guess a chapter or slide number. If no slide genuinely supports a point, leave citation and evidence empty."""


def evaluation_prompt(context: dict, judge: dict, idea: str, slides: str) -> str:
    return f"""You are {judge['name']}, now acting as the judge of a service-innovation workshop. Evaluate the idea
as a rigorous services-marketing expert who applies the course material below, while keeping your customer perspective.

Your customer profile:
{profile_lines(judge['profile'])}

{firm_context(context)}

The idea the team selected for evaluation:
\"\"\"{idea}\"\"\"

Course slides (all decks, full text; each line starts with that slide's label in brackets):
{slides}

{CITATION_RULES}
Judge fit with the firm's mission and the selected innovation strategy, customer value, feasibility, and risks.
Be balanced and specific: list positives, negatives, and concrete suggested changes, then decide GO (the idea can be
executed) or NO GO. If and only if you decide GO and you want evidence of long-term customer value before execution,
request a business analysis (a five-year customer lifetime value / CLV:CAC simulation) and say what it should show."""


def final_decision_prompt(context: dict, judge: dict, idea: str, evaluation: dict, report: str, slides: str) -> str:
    return f"""You are {judge['name']}, the judge of this service-innovation workshop.

{firm_context(context)}

Idea under review:
\"\"\"{idea}\"\"\"

Your earlier evaluation decision: {evaluation['decision']}
Your earlier thoughts: {evaluation['overall_thoughts']}
What you asked the business analysis to show: {evaluation['analysis_request']}

The team has sent you this CLV business analysis report:
{report}

Make the final GO / NO GO decision. A CLV:CAC ratio of about 3:1 or higher is commonly treated as healthy, and a
ratio below 1:1 means customers are worth less than they cost to acquire. Weigh the numbers together with your
earlier evaluation, explain your rationale, and list any conditions or next steps.

Course slides (all decks, full text; each line starts with that slide's label in brackets):
{slides}

{CITATION_RULES}"""


def clv_prompt(context: dict, idea: str, persona: str, profile: dict[str, str]) -> str:
    return f"""Act as a digital twin of a potential customer.

Firm: {context['firm']}
Firm/service description: {context['description']}
New service innovation being launched ({context['innovation_type']}):
{idea}
Target customer persona supplied by the researcher:
{persona}

Individual customer profile:
{profile_lines(profile)}

Estimate this customer's probability (an integer from 0 to 100) of using this new service in
each of the next five years. Years 2-5 are conditional probabilities: estimate the chance of
returning that year if the customer was still active in the prior year.

Be realistically calibrated, not promotional. A generally useful service does not imply 100%
retention. Account for ordinary churn, changing needs, price sensitivity, competing services,
relocation, and declining relevance when consistent with the supplied profile. Reserve values
above 90 for unusually strong, explicit evidence of durable loyalty, and use the full range to
distinguish profiles. Do not invent discounts, product changes, or personal facts. Return five
probabilities and one concise reason."""


def stable_draw(profile_id: int, year: int, firm: str) -> float:
    """Return a reproducible pseudo-random percentile for a profile/year."""
    key = f"{firm.casefold().strip()}|{profile_id}|{year}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return integer / (2**64 - 1) * 100


async def query_profile(profile_id: int, firm: str, prompt: str, llm, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        for attempt in range(5):
            try:
                response = await llm.ainvoke(prompt)
                retained = True
                decisions = {}
                for year in YEARS:
                    probability = int(getattr(response, f"year_{year}_probability"))
                    decisions[f"Year {year} probability"] = probability / 100
                    retained = retained and stable_draw(profile_id, year, firm) < probability
                    decisions[f"Year {year} return"] = retained
                return {"Profile ID": profile_id, **decisions, "Reason": response.reason, "Status": "Success"}
            except Exception as exc:
                if attempt < 4:
                    await asyncio.sleep(2**attempt)
                else:
                    return {"Profile ID": profile_id, "Status": f"Error: {exc}"}
    return {"Profile ID": profile_id, "Status": "Unknown error"}


async def run_all(tasks: list, progress) -> list[dict]:
    results = []
    for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
        results.append(await task)
        progress.progress(completed / len(tasks), text=f"{completed} of {len(tasks)} profiles simulated")
    return results


def simulate_retention(profiles_df: pd.DataFrame, context: dict, idea: str, persona: str, api_key: str) -> pd.DataFrame:
    llm = make_llm(api_key, RetentionResponse, temperature=0.5)
    semaphore = asyncio.Semaphore(40)
    profiles = {}
    tasks = []
    for profile_id, (_, row) in enumerate(profiles_df.iterrows(), start=1):
        profile = {str(column): value for column, raw in row.items() if (value := clean_value(raw)) is not None}
        profiles[profile_id] = profile
        tasks.append(query_profile(
            profile_id, context["firm"], clv_prompt(context, idea, persona, profile), llm, semaphore
        ))
    progress = st.progress(0, text="Preparing profile simulations...")
    responses = asyncio.run(run_all(tasks, progress))
    progress.empty()
    records = [{**profiles[item["Profile ID"]], **item} for item in responses]
    return pd.DataFrame(records).sort_values("Profile ID")


def calculate_clv(results: pd.DataFrame, monthly_sales: float, margin_percent: float) -> pd.DataFrame:
    successful = results.loc[results["Status"] == "Success"]
    annual_sales = monthly_sales * 12
    margin = margin_percent / 100
    return pd.DataFrame([
        {
            "Year": f"Year {year}",
            "Returning profiles": (retained := int(successful[f"Year {year} return"].sum())),
            "Retention rate": retained / len(successful),
            "Annual sales per profile": annual_sales,
            "Margin": margin,
            "Annual CLV contribution": annual_sales * retained * margin,
        }
        for year in YEARS
    ])


def clv_figure(summary: pd.DataFrame, firm: str):
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(summary["Year"], summary["Annual CLV contribution"], color="#2563eb")
    axis.set_title(f"Five-year CLV contribution of the innovation — {firm}", pad=14)
    axis.set_ylabel("CLV contribution ($)")
    axis.grid(axis="y", alpha=0.2)
    axis.bar_label(bars, labels=[f"${value:,.0f}" for value in summary["Annual CLV contribution"]], padding=3)
    figure.tight_layout()
    return figure


def clv_report(results: pd.DataFrame, summary: pd.DataFrame, monthly_sales: float, margin: float, persona: str) -> str:
    successful = results.loc[results["Status"] == "Success"]
    total = float(summary["Annual CLV contribution"].sum())
    average = total / len(successful)
    yearly = "\n".join(
        f"- {row['Year']}: {row['Returning profiles']} returning ({row['Retention rate']:.1%}), "
        f"CLV contribution ${row['Annual CLV contribution']:,.2f}"
        for _, row in summary.iterrows()
    )
    reasons = "\n".join(f"- {reason}" for reason in successful["Reason"].sample(min(8, len(successful)), random_state=1))
    return f"""Target persona: {persona}
Assumptions: ${monthly_sales:,.2f} monthly sales per returning customer, {margin:.0f}% margin, CAC fixed at ${CAC_PER_CUSTOMER:,.0f}, 5-year horizon, no discounting.
Simulated customers: {len(successful)}
5-year cohort CLV: ${total:,.2f}
Average CLV per customer: ${average:,.2f}
CLV : CAC = {average / CAC_PER_CUSTOMER:.2f} : 1
Year by year:
{yearly}
Sample of customer reasons:
{reasons}"""


def cited_text(item: str | dict) -> str:
    if isinstance(item, str):
        return item
    if item["status"] in ("verified", "corrected"):
        return f"{item['point']} *({item['citation']})*"
    if item["status"] == "removed":
        return f"{item['point']} *(slide citation removed: not found in the slides)*"
    return item["point"]


def bullet_list(items: list[str | dict]) -> str:
    return "\n".join(f"- {cited_text(item)}" for item in items) or "- None"


def evidence_list(*groups: list[dict]) -> str:
    quotes = {
        (item["citation"], item["evidence"]) for group in groups for item in group
        if item["status"] in ("verified", "corrected")
    }
    return "\n".join(f"- **{citation}**: “{evidence}”" for citation, evidence in sorted(quotes)) or "- No verified citations"


def workshop_markdown(context: dict, idea: str, evaluation: dict | None, final: dict | None, report: str | None) -> str:
    sections = [f"# Service innovation workshop — {context['firm']}", firm_context(context), f"## Selected idea\n{idea}"]
    if evaluation:
        sections.append(
            f"## Judge evaluation — {evaluation['decision']}\n{evaluation['overall_thoughts']}\n\n"
            f"### Slide concepts applied\n{bullet_list(evaluation['slide_concepts_applied'])}\n\n"
            f"### Positives\n{bullet_list(evaluation['positives'])}\n\n### Negatives\n{bullet_list(evaluation['negatives'])}\n\n"
            f"### Suggested changes\n{bullet_list(evaluation['suggested_changes'])}"
        )
    if report:
        sections.append(f"## CLV business analysis\n{report}")
    if final:
        sections.append(f"## Final decision — {final['decision']}\n{final['rationale']}\n\n{bullet_list(final['conditions'])}")
    return "\n\n".join(sections)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_") or "firm"


def clear_downstream(*keys: str) -> None:
    for key in keys or DOWNSTREAM_KEYS:
        st.session_state.pop(key, None)


def choose_innovation_type(name: str) -> None:
    if st.session_state.get("innovation_type") != name:
        st.session_state["innovation_type"] = name
        clear_downstream()


def use_selected_ideas(ideas: list[str]) -> None:
    chosen = [idea for index, idea in enumerate(ideas) if st.session_state.get(f"idea_pick_{index}")]
    st.session_state["idea_box"] = "\n".join(f"- {idea}" for idea in chosen)


def decision_banner(decision: str, text: str) -> None:
    (st.success if decision == "GO" else st.error)(f"**{decision}** — {text}")


st.set_page_config(page_title="Service Innovation Lab", page_icon="💡", layout="wide")
st.title("Service Innovation Lab")
st.caption(
    "Ten digital-twin customers brainstorm a service innovation, a twin judge evaluates it against the course slides, "
    "and a CLV simulation informs the final GO / NO GO decision."
)

api_key = configured_api_key()
if not api_key:
    st.error("Gemini API key is not configured. Add GOOGLE_API_KEY to the app's Streamlit secrets.")
    st.stop()

st.subheader("1. Firm")
firm_name = st.text_input("Firm name", placeholder="e.g., Acme Fitness")
firm_description = st.text_area(
    "Firm description", height=130,
    placeholder="Describe the firm, its current services, customers, price points, and channels.",
)
firm_mission = st.text_area("Firm mission", height=90, placeholder="Paste or write the firm's mission statement.")

st.subheader("2. Digital-twin panel")
try:
    all_profiles = load_default_profiles().dropna(how="all").reset_index(drop=True)
except Exception as exc:
    st.error(f"Could not load twin profiles: {exc}")
    st.stop()

if "panel" not in st.session_state or st.button("Draw a new random panel of 10 twins"):
    st.session_state["panel"] = select_panel(all_profiles)
    st.session_state["judge"] = random.choice(st.session_state["panel"])["name"]
    clear_downstream()
panel = st.session_state["panel"]
judge = next(member for member in panel if member["name"] == st.session_state["judge"])
st.caption(f"{len(panel)} twins randomly drawn from {len(all_profiles):,} profiles. ⚖️ {judge['name']} will act as the judge.")
st.dataframe(
    pd.DataFrame([{"Twin": member["name"], **member["profile"]} for member in panel]),
    use_container_width=True, hide_index=True,
)

st.subheader("3. Choose the innovation type")
st.caption("Click one cell of the service-innovation matrix (offerings × markets).")
header_blank, header_existing, header_new = st.columns([1, 3, 3])
header_existing.markdown("**Existing customers**")
header_new.markdown("**New customers**")
for offering in ("Existing services", "New services"):
    label_col, *cells = st.columns([1, 3, 3])
    label_col.markdown(f"**{offering}**")
    for cell, market in zip(cells, ("Existing customers", "New customers")):
        name = next(key for key, value in INNOVATION_TYPES.items() if value["offering"] == offering and value["market"] == market)
        selected = st.session_state.get("innovation_type") == name
        cell.button(
            ("✅ " if selected else "") + name, key=f"type_{name}", use_container_width=True,
            type="primary" if selected else "secondary", on_click=choose_innovation_type, args=(name,),
        )
        cell.caption(INNOVATION_TYPES[name]["action"])

innovation_type = st.session_state.get("innovation_type")
if innovation_type:
    st.info(f"Selected: **{innovation_type}** — {INNOVATION_TYPES[innovation_type]['action']}")

context = {
    "firm": firm_name.strip(), "description": firm_description.strip(),
    "mission": firm_mission.strip(), "innovation_type": innovation_type,
}

st.subheader("4. Ideate")
if st.button(f"Start ideating ({DISCUSSION_SECONDS}-second discussion)", type="primary", use_container_width=True):
    missing = [label for label, value in (
        ("firm name", firm_name), ("firm description", firm_description), ("firm mission", firm_mission),
    ) if not value.strip()]
    if missing:
        st.error("Please provide: " + ", ".join(missing) + ".")
    elif not innovation_type:
        st.error("Click one innovation type first.")
    else:
        clear_downstream()
        transcript = run_discussion(context, panel, api_key)
        if not transcript:
            st.error("No twin was able to speak. Check the API key and try again.")
        else:
            try:
                with st.spinner(f"{judge['name']} is collating the discussion..."):
                    collated = collate_ideas(context, judge, transcript, api_key)
                st.session_state["discussion"] = {"transcript": transcript, "context": context}
                st.session_state["collated"] = collated.model_dump()
                st.rerun()
            except Exception as exc:
                st.error(f"Could not collate the discussion: {exc}")

discussion = st.session_state.get("discussion")
collated = st.session_state.get("collated")
if not (discussion and collated):
    st.stop()

session_context = discussion["context"]
spoken = {turn["speaker"] for turn in discussion["transcript"]}
with st.expander(f"Discussion transcript — {len(discussion['transcript'])} turns, {len(spoken)} of {len(panel)} twins spoke", expanded=True):
    for turn in discussion["transcript"]:
        render_turn(turn)

st.subheader(f"5. ⚖️ {judge['name']}'s collated ideas")
st.write(collated["summary"])
st.caption("Tick the points you like, then send them to your idea box. You can edit the box freely.")
for index, idea in enumerate(collated["ideas"]):
    st.checkbox(idea, key=f"idea_pick_{index}")
st.button("Add ticked points to my idea box", on_click=use_selected_ideas, args=(collated["ideas"],))
st.text_area(
    "My idea box (edit, paste, or rewrite the idea you want evaluated)", key="idea_box", height=200,
)
idea_text = st.session_state.get("idea_box", "").strip()
if idea_text:
    st.caption("Copy your idea (use the copy icon at the top right of the box):")
    st.code(idea_text, language=None, wrap_lines=True)

st.subheader("6. Evaluate")
slides, skipped_decks = load_slides()
if skipped_decks:
    st.warning("These slide decks could not be read (close them in PowerPoint/OneDrive and reload): " + "; ".join(skipped_decks))
if st.button("Start evaluating", type="primary", use_container_width=True):
    if not idea_text:
        st.error("Add at least one idea to the idea box first.")
    elif not slides:
        st.error(
            "No slide decks (.pptx) were found. Put them in a folder named `slides` next to the app file. "
            f"Files next to the app ({APP_DIR}): {slide_file_listing()}"
        )
    else:
        try:
            with st.spinner(f"{judge['name']} is evaluating the idea against all course slides..."):
                evaluation = invoke_with_retry(
                    make_llm(api_key, Evaluation, temperature=0.3),
                    evaluation_prompt(session_context, judge, idea_text, slides_for_prompt(slides)),
                )
            st.session_state["evaluation"] = {
                **verify_points(
                    evaluation.model_dump(),
                    ("slide_concepts_applied", "positives", "negatives", "suggested_changes"), slides,
                ),
                "idea": idea_text,
            }
            clear_downstream("clv_innovation", "final_decision")
        except Exception as exc:
            st.error(f"Could not evaluate the idea: {exc}")

evaluation = st.session_state.get("evaluation")
if not evaluation:
    st.stop()
if evaluation["idea"] != idea_text:
    st.warning("The idea box changed after this evaluation. Click **Start evaluating** again to judge the new version.")

st.markdown(f"#### ⚖️ {judge['name']}'s evaluation")
st.write(evaluation["overall_thoughts"])
positive_col, negative_col = st.columns(2)
positive_col.markdown("**👍 Positives**\n" + bullet_list(evaluation["positives"]))
negative_col.markdown("**👎 Negatives**\n" + bullet_list(evaluation["negatives"]))
st.markdown("**🛠️ Suggested changes**\n" + bullet_list(evaluation["suggested_changes"]))
with st.expander("Course concepts the judge applied"):
    st.markdown(bullet_list(evaluation["slide_concepts_applied"]))
with st.expander("Slide evidence behind each citation"):
    st.caption(
        "Every citation shown was checked: its quoted phrase appears on that slide. "
        "Citations whose quote could not be found in any slide were removed."
    )
    st.markdown(evidence_list(*(evaluation[field] for field in (
        "slide_concepts_applied", "positives", "negatives", "suggested_changes"
    ))))
decision_banner(
    evaluation["decision"],
    "the idea can be executed." if evaluation["decision"] == "GO" else "the idea should not be executed as it stands.",
)

report = None
final = st.session_state.get("final_decision")
wants_analysis = evaluation["decision"] == "GO" and evaluation["requests_business_analysis"]
if wants_analysis:
    st.divider()
    st.subheader("7. Business analysis requested")
    st.warning(f"⚖️ {judge['name']}: {evaluation['analysis_request'] or 'Please run a CLV analysis before execution.'}")
    persona = st.text_area(
        "Target customer persona for the innovation", height=160,
        placeholder="Describe needs, behaviors, motivations, pain points, budget, and usage context.",
    )
    sales_col, margin_col = st.columns(2)
    monthly_sales = sales_col.number_input("Expected monthly sales per returning customer ($)", min_value=0.0, value=100.0, step=10.0)
    margin_percent = margin_col.number_input("Profit margin (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    st.info("CAC is fixed at $100 per acquired customer. Monthly sales are annualized (× 12); no discount rate is applied.")
    with st.expander("Customer profiles for the CLV simulation", expanded=False):
        clv_profiles = filter_profiles(all_profiles)
        st.caption(f"{len(clv_profiles):,} of {len(all_profiles):,} profiles selected")

    if st.button("📈 Run CLV analysis", type="primary", use_container_width=True):
        if not persona.strip():
            st.error("Please provide a target customer persona.")
        elif clv_profiles.empty:
            st.error("Select at least one customer profile.")
        else:
            with st.spinner("Simulating five-year repeat use of the innovation..."):
                results = simulate_retention(clv_profiles, session_context, evaluation["idea"], persona, api_key)
            st.session_state["clv_innovation"] = {
                "results": results, "monthly_sales": monthly_sales, "margin": margin_percent, "persona": persona,
            }
            clear_downstream("final_decision")
            final = None

    clv_run = st.session_state.get("clv_innovation")
    if clv_run:
        results = clv_run["results"]
        successful = results.loc[results["Status"] == "Success"]
        if successful.empty:
            st.error("No simulations succeeded. Review the Status column.")
            st.dataframe(results, use_container_width=True)
            st.stop()
        summary = calculate_clv(results, clv_run["monthly_sales"], clv_run["margin"])
        total_clv = float(summary["Annual CLV contribution"].sum())
        average_clv = total_clv / len(successful)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("5-year cohort CLV", f"${total_clv:,.2f}")
        m2.metric("Average CLV per profile", f"${average_clv:,.2f}")
        m3.metric("Total acquisition cost", f"${len(successful) * CAC_PER_CUSTOMER:,.2f}")
        m4.metric("CLV : CAC", f"{average_clv / CAC_PER_CUSTOMER:.2f} : 1")
        figure = clv_figure(summary, session_context["firm"])
        st.pyplot(figure, use_container_width=True)
        image_buffer = BytesIO()
        figure.savefig(image_buffer, format="png", dpi=200, bbox_inches="tight")
        plt.close(figure)
        with st.expander("Profile decisions"):
            st.dataframe(results, use_container_width=True, height=360)
        report = clv_report(results, summary, clv_run["monthly_sales"], clv_run["margin"], clv_run["persona"])
        stem = safe_filename(session_context["firm"])
        d1, d2 = st.columns(2)
        d1.download_button("Download CLV calculation", summary.to_csv(index=False).encode("utf-8-sig"), f"{stem}_innovation_clv.csv", "text/csv")
        d2.download_button("Download chart", image_buffer.getvalue(), f"{stem}_innovation_clv.png", "image/png")

        if st.button(f"📨 Send CLV report to {judge['name']}", type="primary", use_container_width=True):
            try:
                with st.spinner(f"{judge['name']} is reviewing the CLV report..."):
                    decision = invoke_with_retry(
                        make_llm(api_key, FinalDecision, temperature=0.3),
                        final_decision_prompt(
                            session_context, judge, evaluation["idea"], evaluation, report, slides_for_prompt(slides)
                        ),
                    )
                st.session_state["final_decision"] = final = verify_points(decision.model_dump(), ("conditions",), slides)
            except Exception as exc:
                st.error(f"Could not get the final decision: {exc}")

        if final:
            st.subheader(f"8. ⚖️ {judge['name']}'s final decision")
            decision_banner(final["decision"], final["rationale"])
            st.markdown("**Conditions and next steps**\n" + bullet_list(final["conditions"]))
            with st.expander("Slide evidence behind the final decision"):
                st.markdown(evidence_list(final["conditions"]))
elif evaluation["decision"] == "GO":
    st.caption(f"{judge['name']} did not request a business analysis; the GO decision stands.")
else:
    st.caption("Revise the idea in the idea box using the suggested changes and evaluate again.")

st.divider()
st.download_button(
    "Download workshop report (Markdown)",
    workshop_markdown(session_context, evaluation["idea"], evaluation, final, report).encode("utf-8"),
    f"{safe_filename(session_context['firm'])}_service_innovation.md",
    "text/markdown",
)
