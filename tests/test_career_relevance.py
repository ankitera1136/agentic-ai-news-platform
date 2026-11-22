"""career_relevance() must pass real layoff/hiring titles and reject known false positives."""
import pytest

from backend.services.career.filters import FILTER_VERSION, career_relevance

# Real titles seen in the feeds checked for DATA_PLAN.md (2026-11-15).
REAL_TITLES = [
    "Qualcomm layoffs raise questions over severance support for affected employees in India",
    "Qualcomm Begins Layoffs In India; 1,000 Jobs At Risk As Chipmaker Restructures Global Workforce",
    "PVH to cut around 1,040 jobs globally, targeting office-based roles",
    "Innoviz Technologies to cut 20% of workforce",
    "Tencent denies plans to cut half its workforce, says AI restructuring remarks were misinterpreted",
    "Qantas plans to cut 850 jobs",
    "Boeing slows hiring, plans job cuts as recovery takes longer than expected",
    "Meta to cut one-third of France workforce through voluntary buyouts?",
    "Will Canva let go 1,271 from its workforce?",
    "NICE cuts 500 jobs as AI reshapes customer-service business",
    "Canva lays off over 1,200 people amid slowing growth",
    "Web host Squarespace lays off 19% of its employees",
    "Qualcomm employee laid off days before wedding; what his manager told him left him stunned",
    "PhysicsWallah cites AI efficiencies for job cuts; offline biz faces profit pressure",
    "Indian GCCs set to add 35k jobs in 2027",
    "GCCs ramp up hiring for AI, data, cybersecurity professionals; face talent shortage: Report",
    "Semiconductor boom may add 1 lakh jobs; senior engineers earn 6x IT salary",
    "Fintech giant Plaid opening Seattle engineering office, with plans to double local headcount",
    "Autonomous (YC F25) is hiring MTS to build AI agents that manage your wealth",
    "Disney Layoffs: Entertainment Giant Eyes Thousands Of Job Cuts, Targets 20% Cost Savings",
    "Australian companies face 'AI washing' claims after wave of job cuts, offshoring",
    # v2: the three recall misses found in group 4
    "Microsoft offers voluntary exit packages to 60,000 eligible employees, including in India",
    "Google extends voluntary buyout offers to employees in several countries",
    "Qantas AI Jobs are Coming to India as Outsourcing Deal Affects 850 Roles",
    # v2: other phrasings of the new patterns
    "Telco launches voluntary separation programme for 4,000 staff",
    "Steelmaker says 1,200 jobs at risk after plant closure",
    "850 roles impacted by restructuring at bank",
    "Airline plans offshoring of 400 roles to Manila",
]

# Known false positives from the current DB and the feed checks.
FALSE_POSITIVES = [
    "The OpenAI board member who fired Sam Altman says we still can&#8217;t trust AI CEOs",
    "How do you hire a human? Q&A with Tom Harvey, Stack Overflow technical recruiter",
    "A family tree of software job titles",
    "China reportedly recruited AI exec to spy on Jensen Huang",
    "Nvidia approaches Arm Holdings CEO Haas to take over Jensen Huang's job, Elon Musk backs Huang's reinstatement",
    "Nvidia hires major US law firm to investigate alleged Chinese spy affair",
    "Who is John Langford? Meet the mind behind AI Frontiers who quit after Microsoft's restructuring",
    "Tata Motors PV posts Q2 consolidated loss on JLR restructuring charges",
    "Andhra Pradesh targets 50% women's workforce participation",
    "Why professionals keep applying for jobs they don't want",
    "How a civil servant got away with juggling five jobs for so long",
    "Ajinkya Rahane Picks Up Stake In D2C Cookware Brand Ember",
    "Crows can evaluate visual info twice as fast as humans",
    "Lutnick Pushed to Weaken Chip Rules That Overlap with His Family Firm",
    # v2 patterns must not over-match
    "Nvidia CEO's exit package revealed after board vote",
    "Microsoft outage affects 20 million users worldwide",
    "Cloud outage affects 74 countries",
    "Apple cuts employee AI budgets in cost-control push: Report",
    "Kutlerri.ai raises $4 Mn from Narahari Investments to expand AI solutions for restaurants",
    "Voluntary donations to open-source projects hit a record",
    "Study of 850 roles in video games finds gender gap",
]


@pytest.mark.parametrize("title", REAL_TITLES)
def test_real_career_titles_pass(title):
    assert career_relevance(title, "") is not None, title


@pytest.mark.parametrize("title", FALSE_POSITIVES)
def test_known_false_positives_are_rejected(title):
    assert career_relevance(title, "") is None, title


def test_reason_names_the_pattern():
    assert career_relevance("Acme lays off 300 staff") == "layoff"


def test_summary_is_checked_when_title_is_neutral():
    reason = career_relevance("Acme Q3 update", "<p>The company will <b>lay off</b> 200 people.</p>")
    assert reason == "summary:layoff"


def test_html_entities_and_tags_in_summary_are_cleaned():
    assert career_relevance("Update", "<a href='x'>fired Sam Altman</a> &amp; more") is None


def test_empty_and_none_inputs():
    assert career_relevance(None, None) is None
    assert career_relevance("", "") is None
    assert career_relevance("Acme lays off 10 people", None) == "layoff"


def test_filter_version_is_v2():
    assert FILTER_VERSION == "career-filter-v2"
