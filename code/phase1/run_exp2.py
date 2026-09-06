"""
Experiment 2: Baseline Extension — n=150
==========================================
Extends the §8 baseline from n=50 to n=150 (50 prompts per domain).
Generates 100 new prompts following the same protocol as the original 50,
then scores all 150 with Gemini 2.5 Flash single-pass.

New prompts distribution (100 total):
  - Consulting: 30 new (9 UV / 12 Med / 9 Det)  → 50 total (20+30)
  - Medical:    35 new (10 UV / 14 Med / 11 Det) → 50 total (15+35)
  - Payments:   35 new (10 UV / 14 Med / 11 Det) → 50 total (15+35)

Overall vagueness: ~30% Ultra-vague / ~40% Medium / ~30% Detailed

Output:
  prompts_new_100.csv              — the 100 new prompts
  baseline_extended_n150.csv       — all 150 scores (prompt_id, domain, vagueness, cov, scorer)
  baseline_extended_stats.json     — mean/std/CI by domain and vagueness
  table_baseline_latex.tex         — LaTeX table for paper §8

Author: (anonymised for review)
Date: 2026-06-30
Seed: 42
"""

import json
import csv
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))  # [relocated]
from scoring import score_response, get_gemini_client, call_gemini

# ─── Constants ────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "recomputed"  # [relocated]
OUTPUT_DIR.mkdir(exist_ok=True)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "data"  # [relocated]

SINGLE_PASS_SYSTEM = (
    "You are a project requirements analyst. Extract all requirements "
    "from the following description and structure them into a complete, "
    "detailed specification. Be thorough — cover every aspect needed "
    "to implement this project/configure this system."
)

# ─── New prompts (100, hardcoded for reproducibility) ─────────────────────────

NEW_PROMPTS = [
    # ── Consulting Ultra-vague (9) ─────────────────────────────────────────
    {"id": "N01", "domain": "Consulting", "vagueness": "Ultra-vague",
     "text": "I want to build a mobile app for my business."},
    {"id": "N02", "domain": "Consulting", "vagueness": "Ultra-vague",
     "text": "I need a chatbot on my website to handle customer questions."},
    {"id": "N03", "domain": "Consulting", "vagueness": "Ultra-vague",
     "text": "I want to create a blog and grow an audience online."},
    {"id": "N04", "domain": "Consulting", "vagueness": "Ultra-vague",
     "text": "Help me build an API for my application."},
    {"id": "N05", "domain": "Consulting", "vagueness": "Ultra-vague",
     "text": "I want a booking system for my service business."},
    {"id": "N06", "domain": "Consulting", "vagueness": "Ultra-vague",
     "text": "I need a CRM tool for my small sales team."},
    {"id": "N07", "domain": "Consulting", "vagueness": "Ultra-vague",
     "text": "I want to redesign my company website, it looks outdated."},
    {"id": "N08", "domain": "Consulting", "vagueness": "Ultra-vague",
     "text": "I need an analytics dashboard to track my business metrics."},
    {"id": "N09", "domain": "Consulting", "vagueness": "Ultra-vague",
     "text": "Help me launch a newsletter for my audience."},

    # ── Consulting Medium (12) ─────────────────────────────────────────────
    {"id": "N10", "domain": "Consulting", "vagueness": "Medium",
     "text": "I'm building an HR tool for remote teams. It needs to handle time-off requests, employee onboarding checklists, and an org chart. Should be accessible from mobile."},
    {"id": "N11", "domain": "Consulting", "vagueness": "Medium",
     "text": "I want a fitness coaching app. Users should get personalized workout plans, track progress with charts, log meals, and subscribe monthly. Trainer dashboard needed."},
    {"id": "N12", "domain": "Consulting", "vagueness": "Medium",
     "text": "I need a real estate listing website. Visitors can search by city, price, and property type. Each listing has photos, a map, and a contact form for the agent."},
    {"id": "N13", "domain": "Consulting", "vagueness": "Medium",
     "text": "Our company needs a corporate intranet. Employees should find documents, see the team directory, read announcements, and submit IT helpdesk tickets."},
    {"id": "N14", "domain": "Consulting", "vagueness": "Medium",
     "text": "I'm building a tutoring marketplace. Students post their needs, tutors apply, they schedule video sessions through the platform, and payment is handled in-app with a 15% commission."},
    {"id": "N15", "domain": "Consulting", "vagueness": "Medium",
     "text": "I want an events ticketing site for live music. Organisers create events, set seat maps, and attendees buy tickets with Stripe. QR code entry at the door. Email confirmations."},
    {"id": "N16", "domain": "Consulting", "vagueness": "Medium",
     "text": "I need a travel agency website with searchable tour packages, online booking requests, a blog with destination guides, and a testimonials section."},
    {"id": "N17", "domain": "Consulting", "vagueness": "Medium",
     "text": "I'm launching a job board for tech startups. Companies post roles, candidates apply with their LinkedIn or a CV upload, and hiring managers get an applicant dashboard. Email alerts for new applications."},
    {"id": "N18", "domain": "Consulting", "vagueness": "Medium",
     "text": "I need an expense management tool for my 20-person company. Employees upload receipts, their manager approves, and the finance team exports to Excel. Monthly budget reports per team."},
    {"id": "N19", "domain": "Consulting", "vagueness": "Medium",
     "text": "I want a delivery tracking app for my local courier business. Drivers have a mobile app to mark deliveries done and capture signatures. Customers get an SMS link to track their parcel in real time."},
    {"id": "N20", "domain": "Consulting", "vagueness": "Medium",
     "text": "I'm building an online coding bootcamp platform. I need video lessons, interactive coding exercises, quizzes, a leaderboard, and a certificate generator when students complete a module."},
    {"id": "N21", "domain": "Consulting", "vagueness": "Medium",
     "text": "I want to build a community forum for indie game developers. Users create topics, vote on posts, showcase their games with screenshots, and get badges for contributions. Moderation tools needed."},

    # ── Consulting Detailed (9) ────────────────────────────────────────────
    {"id": "N22", "domain": "Consulting", "vagueness": "Detailed",
     "text": "I'm launching a B2C subscription box for organic skincare. Monthly and quarterly plans at €45/€120. Customers take a skin-type quiz at signup to personalise boxes. Referral program: €10 credit per referred friend who subscribes. Shopify storefront, Recharge for subscriptions, Klaviyo for email flows. Need a dashboard showing churn rate, LTV, and box contents performance. Target: French market first."},
    {"id": "N23", "domain": "Consulting", "vagueness": "Detailed",
     "text": "I need a patient portal for our private clinic. Patients can message their doctor securely, book appointments, view lab results and prescriptions, and fill pre-consultation forms. Must be HIPAA-compliant (EU equivalent: RGPD healthcare). Two-factor authentication mandatory. Mobile-first. Integration with our existing Doctolib calendar and HL7 FHIR for lab data import."},
    {"id": "N24", "domain": "Consulting", "vagueness": "Detailed",
     "text": "I'm building a SaaS for restaurant chains with 10+ locations. Features: centralised inventory management with automatic low-stock alerts, staff scheduling with shift swaps, daily sales sync from POS (Toast and Lightspeed), and a GM performance dashboard. Multi-tenant architecture so each chain has isolated data. Pricing: €199/location/month. Launch in 6 months."},
    {"id": "N25", "domain": "Consulting", "vagueness": "Detailed",
     "text": "We're building a B2B procurement platform for mid-sized manufacturers. Buyers request quotes from multiple suppliers, compare responses, and approve POs with a 3-step approval workflow. Integration with SAP ERP via REST API. Suppliers have a vendor portal. All documents (POs, invoices, delivery notes) stored with version control. GDPR compliant. Audit log for all actions. SOC 2 Type II certification required."},
    {"id": "N26", "domain": "Consulting", "vagueness": "Detailed",
     "text": "I need a fintech dashboard for retail investors. Real-time portfolio tracking with P&L, allocation breakdown by sector/geography/asset class. Integrated news feed filtered by holdings. Stock screener with 30+ filters. Watchlist with price alerts. Brokerage integration: Interactive Brokers and Degiro via OAuth. User authentication with biometrics on mobile. Free tier (3 portfolios) and Pro at €15/month (unlimited). GDPR, no data selling."},
    {"id": "N27", "domain": "Consulting", "vagueness": "Detailed",
     "text": "Our logistics company needs a platform for SME importers. Features: freight quote aggregation from 5 carriers (API integration), real-time shipment tracking, automated customs document generation (commercial invoice, packing list, bill of lading), HS code lookup, and a compliance alert system for embargoed goods. Multi-currency: EUR, USD, CNY. Customer-facing portal + internal ops dashboard. Target: 200 SME clients in Year 1."},
    {"id": "N28", "domain": "Consulting", "vagueness": "Detailed",
     "text": "I'm launching a marketplace for professional services (lawyers, accountants, HR consultants) in France. Professionals create verified profiles (SIRET check, bar association number). Clients browse, read reviews, and book consultations. Payment in escrow (Mangopay), released after service confirmed. Dispute resolution workflow. SEO is critical — we need structured data markup for all profiles. Commission: 12% per booking. Legal compliance: professional liability insurance verification."},
    {"id": "N29", "domain": "Consulting", "vagueness": "Detailed",
     "text": "Our edtech startup needs a K-12 learning platform for private schools. Teacher dashboard to create assignments, track completion, and grade. Student interface with gamified progress (XP, badges, leaderboard). Parent portal to follow child's progress and message teachers. Curriculum aligned to French national curriculum (CP to Terminale). GDPR for minors (parental consent flows). LTI integration with existing school SIS (Pronote). Pricing: B2B annual license per school (€2,000-€20,000 depending on size)."},
    {"id": "N30", "domain": "Consulting", "vagueness": "Detailed",
     "text": "I need a complete digital marketing infrastructure for a D2C cosmetics brand launching in France and Germany. This includes: Shopify storefront (bilingual FR/DE), Google Ads and Meta Ads tracking with server-side events (GA4 + Meta CAPI), a post-purchase email and SMS sequence (Klaviyo), influencer affiliate tracking (Impact.com), a loyalty program (points per purchase, tiered rewards), and a monthly performance dashboard showing ROAS, CAC, LTV by channel. Budget tracking: €15k/month ad spend, target CAC < €25, LTV/CAC > 4."},

    # ── Medical Ultra-vague (10) ───────────────────────────────────────────
    {"id": "N31", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "Patient has prostate cancer. On hormone therapy. Seems to be doing fine."},
    {"id": "N32", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "Got a referral for a patient with possible lymphoma. Biopsy not done yet."},
    {"id": "N33", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "Post-op patient, head and neck. Need to plan follow-up."},
    {"id": "N34", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "Young patient, suspected leukemia. Parents very anxious. Need next steps."},
    {"id": "N35", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "Patient finished radiation last month. Now in surveillance. Everything looks okay."},
    {"id": "N36", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "Elderly patient with colon cancer. He doesn't want aggressive treatment."},
    {"id": "N37", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "Patient on myeloma treatment. Bone pain is better. Monthly infusions ongoing."},
    {"id": "N38", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "New patient, thyroid cancer, very early stage. Labs all normal. What do we do?"},
    {"id": "N39", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "Patient has a brain tumor. Currently stable on steroids. Follow-up needed."},
    {"id": "N40", "domain": "Medical", "vagueness": "Ultra-vague",
     "text": "Patient diagnosed with ovarian cancer. Just told the family. Next steps unclear."},

    # ── Medical Medium (14) ────────────────────────────────────────────────
    {"id": "N41", "domain": "Medical", "vagueness": "Medium",
     "text": "55-year-old male, stage III colon cancer (T3N2M0). Completed 8 cycles of FOLFOX adjuvant chemotherapy 4 months ago. Now starting surveillance: CT every 6 months, CEA monthly. No current symptoms."},
    {"id": "N42", "domain": "Medical", "vagueness": "Medium",
     "text": "Patient with EGFR-mutant NSCLC (exon 19 deletion), stage IVB, on osimertinib 80mg daily. 6-month restaging CT shows stable disease, no new lesions. Tolerating treatment well, mild diarrhea grade 1."},
    {"id": "N43", "domain": "Medical", "vagueness": "Medium",
     "text": "48-year-old female, cervical cancer stage IB2, HPV-positive. Planning concurrent chemoradiation: cisplatin weekly + pelvic EBRT (45 Gy/25 fractions) followed by brachytherapy boost. No prior treatment."},
    {"id": "N44", "domain": "Medical", "vagueness": "Medium",
     "text": "Patient post-allogeneic stem cell transplant for AML, day +120. Developing signs of mild skin GVHD (grade I). Currently on tacrolimus and methylprednisolone. CMV monitoring negative. Chimerism 98%."},
    {"id": "N45", "domain": "Medical", "vagueness": "Medium",
     "text": "67-year-old male, urothelial carcinoma, underwent radical cystectomy 6 months ago (pT3N1). Started adjuvant nivolumab 3 months ago. Recent CT: no evidence of recurrence. PSA undetectable. Tolerating immunotherapy."},
    {"id": "N46", "domain": "Medical", "vagueness": "Medium",
     "text": "35-year-old female, classical Hodgkin lymphoma stage IIA (cervical + mediastinal nodes). Mid-treatment PET after 2 cycles ABVD shows Deauville score 2 (complete metabolic response). Planning 2 more cycles then restaging."},
    {"id": "N47", "domain": "Medical", "vagueness": "Medium",
     "text": "Patient with metastatic melanoma (BRAF V600E mutated), on pembrolizumab 200mg Q3W. After cycle 4, developing immune-related colitis grade 2 with diarrhea 4-6x/day. Stool calprotectin elevated."},
    {"id": "N48", "domain": "Medical", "vagueness": "Medium",
     "text": "71-year-old female with CLL (Rai stage III), started ibrutinib 420mg daily 4 months ago. Recent CBC: WBC down from 142k to 38k, Hb stable at 10.2g/dL. No significant side effects. Surveillance CT scheduled."},
    {"id": "N49", "domain": "Medical", "vagueness": "Medium",
     "text": "7-year-old boy, newly diagnosed Wilms tumor (right kidney, ~8cm). Staging CT shows no pulmonary metastases, left kidney normal. Planning pre-operative chemotherapy (vincristine + actinomycin-D) per SIOP protocol."},
    {"id": "N50", "domain": "Medical", "vagueness": "Medium",
     "text": "60-year-old male, 8 weeks post-esophagectomy for adenocarcinoma (pT3N1). Struggling with nutrition: weight loss 8kg since surgery. Currently on jejunostomy feeds. Starting oral diet trials. Pathology showed R0 resection, 2/15 nodes positive."},
    {"id": "N51", "domain": "Medical", "vagueness": "Medium",
     "text": "52-year-old female, recurrent ovarian cancer (high-grade serous, BRCA1 mutated), platinum-sensitive relapse after 14-month remission. Considering carboplatin + paclitaxel followed by PARP inhibitor maintenance (olaparib)."},
    {"id": "N52", "domain": "Medical", "vagueness": "Medium",
     "text": "44-year-old male, oropharyngeal cancer (HPV+), p16+, T2N2bM0. Completed definitive chemoradiation (70 Gy/35 fractions, cisplatin 100mg/m² Q3W). 3-month post-treatment PET/CT scheduled. Currently managing xerostomia and dysphagia."},
    {"id": "N53", "domain": "Medical", "vagueness": "Medium",
     "text": "78-year-old male, multiple myeloma, on lenalidomide maintenance after ASCT (18 months). Creatinine rising (1.8 → 2.3 mg/dL over 3 months). M-protein stable. Considering lenalidomide dose reduction."},
    {"id": "N54", "domain": "Medical", "vagueness": "Medium",
     "text": "63-year-old female, endometrial cancer stage IA (grade 1, endometrioid). Post-hysterectomy with bilateral salpingo-oophorectomy. Negative nodes. LVSI absent. No adjuvant therapy recommended per ESMO guidelines. Surveillance every 6 months."},

    # ── Medical Detailed (11) ──────────────────────────────────────────────
    {"id": "N55", "domain": "Medical", "vagueness": "Detailed",
     "text": "58-year-old male. NSCLC adenocarcinoma, stage IVA (T3N2M1a — contralateral lobe). PD-L1 TPS 90%, EGFR/ALK/ROS1 wild-type. Started pembrolizumab 200mg Q3W 3 weeks ago. CBC: WBC 8.2, Hb 10.4 (mild anemia), platelets 210k. No significant comorbidities. Baseline CT: 4.2cm primary mass, 2 contralateral pulmonary nodules < 1cm. No liver or brain mets. First follow-up assessment requested."},
    {"id": "N56", "domain": "Medical", "vagueness": "Detailed",
     "text": "66-year-old female, locally advanced pancreatic ductal adenocarcinoma (T4N1M0, borderline resectable per NCCN). Completed 4 cycles FOLFIRINOX. CA 19-9 decreased from 2,450 to 830 U/mL. Imaging: primary tumor reduced 4.8cm → 3.5cm, SMV involvement improved. Planning restaging CT and multidisciplinary surgical reassessment. On warfarin for permanent AFib (INR 2.1), metformin 1g BD for T2DM. No dose reductions during chemo."},
    {"id": "N57", "domain": "Medical", "vagueness": "Detailed",
     "text": "42-year-old female, invasive ductal carcinoma grade 3, ER+ HER2+, T2N1M0, diagnosed at 28 weeks gestation. MDT decision: start neoadjuvant AC chemotherapy (doxorubicin 60mg/m² + cyclophosphamide 600mg/m² Q3W × 4 cycles) — safe from second trimester. Trastuzumab deferred until post-delivery due to fetal risk. Delivery planned at 37 weeks via C-section. Obstetrics involved. Patient highly anxious, requesting psychological support. Partner present at all consultations."},
    {"id": "N58", "domain": "Medical", "vagueness": "Detailed",
     "text": "70-year-old male, glioblastoma multiforme (IDH wild-type, MGMT promoter unmethylated, TERT mutated). Gross total resection 8 weeks ago. Completed Stupp protocol: temozolomide 75mg/m² daily concurrent with 60 Gy/30 fractions EBRT. MRI at 6 weeks post-RT: no clear progression (pseudoprogression vs. residual). Starting adjuvant temozolomide 200mg/m² D1-5 Q28 days. ECOG 1. Lives with wife. Driving restrictions in place (epilepsy risk — on levetiracetam 1g BD)."},
    {"id": "N59", "domain": "Medical", "vagueness": "Detailed",
     "text": "55-year-old male, gastric adenocarcinoma (HER2+ 3+, MSI-H), stage IV with peritoneal carcinomatosis and 2 liver mets. First-line nivolumab 360mg + FOLFOX Q3W: 3 cycles completed. CT shows partial response in gastric primary but new small-volume ascites. ECOG performance status declining: 1 → 2. Albumin 2.9 g/dL. Patient requesting to continue aggressive treatment. PCI score not formally assessed. MDT discussion ongoing regarding second-line options (ramucirumab + paclitaxel)."},
    {"id": "N60", "domain": "Medical", "vagueness": "Detailed",
     "text": "29-year-old female, AML (FLT3-ITD positive, NPM1 mutated). Induction 7+3 achieved complete remission (MRD negative by PCR). Consolidation: 2 cycles HiDAC (3g/m² Q12h × 6 doses) completed. Evaluating allogeneic SCT vs. continued HiDAC consolidation. Sibling donor: 8/8 HLA match identified. Patient reluctant to proceed with SCT due to employment concerns. Ferritin 2,850 μg/L. LFTs: ALT 68 U/L (mild elevation). Menstrual cycle disrupted — fertility counselling requested."},
    {"id": "N61", "domain": "Medical", "vagueness": "Detailed",
     "text": "68-year-old male, renal cell carcinoma (clear cell, VHL mutated, IMDC intermediate risk), stage IV with pulmonary mets (5 lesions, largest 1.8cm). Progressed on sunitinib after 14 months (best response: stable disease). Now 2 months into second-line cabozantinib 60mg daily. Side effects: hypertension (managed with amlodipine), palmar-plantar erythrodysesthesia grade 2 (no dose reduction yet). 3-month restaging CT planned. PSA 1.2 (irrelevant). Baseline functional status ECOG 1."},
    {"id": "N62", "domain": "Medical", "vagueness": "Detailed",
     "text": "50-year-old female, squamous cell carcinoma of the anal canal, T2N1M0 (HPV+). Planned treatment: definitive chemoradiation — 50.4 Gy in 28 fractions to pelvis + 54 Gy boost to primary (IMRT technique), concurrent mitomycin-C 10mg/m² D1 + 5-FU 1000mg/m²/day CI D1-4 and D29-32. HIV-positive: undetectable viral load, CD4 count 620 cells/μL, on bolutegravir-based ART. Infectious disease co-management. Concern: radiation-related immunosuppression and ART interactions."},
    {"id": "N63", "domain": "Medical", "vagueness": "Detailed",
     "text": "77-year-old male, CLL with confirmed Richter transformation to DLBCL (diffuse large B-cell, non-GCB subtype) on lymph node excision biopsy. Background: Rai stage IV CLL on venetoclax for 18 months prior. ECOG 3. Significant cardiac history: permanent pacemaker, LVEF 38%, NYHA class II. R-CHOP deemed too cardiotoxic. Options discussed: R-miniCHOP vs. polatuzumab vedotin + BR (pola-BR) vs. best supportive care. Goals of care conversation documented. Patient expresses wish to remain at home. Family involved."},
    {"id": "N64", "domain": "Medical", "vagueness": "Detailed",
     "text": "61-year-old female, triple-negative breast cancer (BRCA2 germline mutation found incidentally), stage IIB (T2N1). Neoadjuvant carboplatin AUC5 + paclitaxel 80mg/m² weekly ×12: partial response (pCR not achieved). Post-mastectomy pathology: ypT2ypN1mi. Adjuvant capecitabine recommended (CREATE-X data — 8 cycles, 1250mg/m² BD D1-14 Q21). Genetic counselling referral placed. Patient premenopausal, 1 child, asking about fertility preservation (too late for oocyte cryopreservation pre-chemo). Lymphedema developing — physiotherapy started."},
    {"id": "N65", "domain": "Medical", "vagueness": "Detailed",
     "text": "MDT discussion: 45-year-old male, well-differentiated neuroendocrine tumor (grade 2, Ki-67 8%), jejunal primary with extensive liver metastases (>10 lesions, bilobar). 68Ga-DOTATATE PET/CT: high somatostatin receptor expression (Krenning 3-4). Surgical debulking not recommended given extent. 14 months on lanreotide 120mg Q4W: stable disease (mRECIST). PRRT eligibility confirmed (renal function GFR 72 mL/min, bone marrow reserve adequate). Planning Lu-177 dotatate (4 cycles, 7.4 GBq each, Q8W). Patient informed of fatigue and delayed haematotoxicity risk."},

    # ── Payments Ultra-vague (10) ──────────────────────────────────────────
    {"id": "N66", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "I want to get into DeFi but I don't really know where to start. Don't lose my money."},
    {"id": "N67", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "Help me set up a way to receive money from clients online for my freelance work."},
    {"id": "N68", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "I want to automate my savings and maybe invest some of it."},
    {"id": "N69", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "Set up something that trades for me while I sleep. Nothing too risky."},
    {"id": "N70", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "I want to get into NFT trading. Keep it simple and don't spend too much."},
    {"id": "N71", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "I need to accept crypto payments for my small online shop."},
    {"id": "N72", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "I want to move money automatically between my accounts when certain conditions are met."},
    {"id": "N73", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "I have some idle cash sitting in my bank. Help me make it work a bit, safely."},
    {"id": "N74", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "I want to trade commodities with an automated bot. Standard risk."},
    {"id": "N75", "domain": "Payments", "vagueness": "Ultra-vague",
     "text": "I need a payment solution for my new marketplace app."},

    # ── Payments Medium (14) ───────────────────────────────────────────────
    {"id": "N76", "domain": "Payments", "vagueness": "Medium",
     "text": "I want to set up dollar-cost averaging for crypto: $500/week, split 60% BTC and 40% ETH, executed automatically on Coinbase. Stop if total portfolio drops more than 30% from all-time high."},
    {"id": "N77", "domain": "Payments", "vagueness": "Medium",
     "text": "I'm building a subscription box e-commerce. I need Stripe Billing with monthly and annual plans, proration when upgrading, automatic dunning for failed payments (3 retries over 2 weeks), and pause subscription option."},
    {"id": "N78", "domain": "Payments", "vagueness": "Medium",
     "text": "I want a limit-order-only trading account for SPY options. Max 5 contracts per trade, never hold options overnight, no trades in the first or last 30 minutes of the session. Total capital: $20,000."},
    {"id": "N79", "domain": "Payments", "vagueness": "Medium",
     "text": "My company pays 50+ contractors in 15 countries monthly. I need to minimise FX fees, handle local tax withholding where required, and keep a full audit trail of every transfer. Wire and local bank options needed."},
    {"id": "N80", "domain": "Payments", "vagueness": "Medium",
     "text": "I want a yield farming strategy on Curve Finance: $30k into the FRAX/USDC pool, auto-compound rewards daily, and auto-exit if APY drops below 5% for 3 consecutive days."},
    {"id": "N81", "domain": "Payments", "vagueness": "Medium",
     "text": "I need a tipping system for my creator platform (like Twitch). Users tip in USD, the creator receives 85% after platform fees. Creators can withdraw instantly to their Stripe connected account, minimum $10 payout."},
    {"id": "N82", "domain": "Payments", "vagueness": "Medium",
     "text": "I want to trade gold futures (GC contract) automatically. Max 2 contracts at a time, stop loss at 0.5% per trade, take profit at 1.5%, max 3 trades per day. Only trade US session (14:30–21:00 UTC). Broker: Interactive Brokers."},
    {"id": "N83", "domain": "Payments", "vagueness": "Medium",
     "text": "I need a multi-currency business account. I receive payments in USD, GBP, EUR, and AED. I want to auto-convert everything to EUR at month-end at the best available rate. No conversion during weekends."},
    {"id": "N84", "domain": "Payments", "vagueness": "Medium",
     "text": "I want to run a staking-as-a-service product. Users deposit ETH, we stake with validators via Lido, charge a 10% fee on staking rewards, and send weekly payouts to users automatically. Minimum deposit: 0.1 ETH."},
    {"id": "N85", "domain": "Payments", "vagueness": "Medium",
     "text": "I need a recurring billing system for my gym chain. Monthly membership €45, annual €480, family plan €120/month. Members can pause up to 2 months per year. Failed payment retries with SMS notification."},
    {"id": "N86", "domain": "Payments", "vagueness": "Medium",
     "text": "I want to hedge my BTC holdings against downside. I hold 2 BTC and can tolerate a max 20% loss. Set up protective puts on a regulated derivatives exchange. Auto-roll monthly if still in position."},
    {"id": "N87", "domain": "Payments", "vagueness": "Medium",
     "text": "I need an automated wire transfer rule: every Friday at 17:00 UTC, move any balance above €10,000 from my business account to my savings account, leaving exactly €5,000 in the business account."},
    {"id": "N88", "domain": "Payments", "vagueness": "Medium",
     "text": "I want to manage liquidity across three stablecoin pools on Uniswap V3 (USDC/USDT, USDC/DAI, USDT/DAI). Rebalance daily to maintain equal allocation. Target 0.05% fee tier. Auto-compound earned fees."},
    {"id": "N89", "domain": "Payments", "vagueness": "Medium",
     "text": "I need to build a charity donation platform. Accept credit card and crypto (BTC + ETH). Generate tax receipts automatically. Donors can set up recurring monthly donations. Funds disbursed quarterly to partner NGOs after board approval."},

    # ── Payments Detailed (11) ─────────────────────────────────────────────
    {"id": "N90", "domain": "Payments", "vagueness": "Detailed",
     "text": "Configure a fixed-income automated strategy for my $200k portfolio. Allocation: 60% US T-bills (3-6 month, via TreasuryDirect API, auto-roll at maturity), 40% investment-grade corporate bond ETF (LQD). Quarterly rebalancing. No leverage. Auto-reinvest all dividends and coupon payments. Monthly PDF report for my accountant showing yield, duration, and unrealised P&L. Custodian: Schwab. Tax-optimised for US resident (long-term capital gains preference)."},
    {"id": "N91", "domain": "Payments", "vagueness": "Detailed",
     "text": "I'm building a P2P lending platform for SMEs in Southeast Asia. Loan amounts $1,000–$50,000. Terms: 30/60/90 days. Repayment via local bank transfer or e-wallet (GCash Philippines, PromptPay Thailand). Platform fee: 2% origination + 1% monthly servicing. Credit scoring via bank statement API (Plaid equivalent for SEA). Investor dashboard: portfolio breakdown, expected returns, defaults. Registered in Singapore (MAS fintech sandbox). AML/KYC: Sumsub integration."},
    {"id": "N92", "domain": "Payments", "vagueness": "Detailed",
     "text": "Set up a forex trading bot for EUR/USD, GBP/USD, and USD/JPY. Strategy: 20/50 SMA crossover. Only trade London/New York overlap (13:00–17:00 UTC). Position size: 1% of account per trade ($150 on $15k account). Stop-loss: 30 pips, take-profit: 60 pips. Max 3 open positions simultaneously. Broker: OANDA API v20. No trading on major economic announcement days (NFP, CPI, central bank decisions). Telegram alert for every trade open/close."},
    {"id": "N93", "domain": "Payments", "vagueness": "Detailed",
     "text": "I need a crypto treasury management system for my startup. Holdings: BTC, ETH, USDC. Rules: maintain minimum 6-month runway in USDC (currently $180k). Auto-convert incoming ETH/BTC to USDC if their price drops below the 30-day moving average. All transactions above $50k require multi-sig approval (Gnosis Safe, 2-of-4 signers: CEO, CFO, and 2 board members). Monthly CFO report: unrealised P&L, runway estimate, BTC/ETH exposure. Integrate with accounting system (Xero)."},
    {"id": "N94", "domain": "Payments", "vagueness": "Detailed",
     "text": "Build a statistical arbitrage bot across Binance, Kraken, and Coinbase Pro targeting BTC/USDT price discrepancies > 0.15%. Max capital deployed per arb: $5,000. Full round-trip must complete within 500ms. Risk controls: kill switch if 3 failed attempts in 5 minutes, max daily loss $2,000, Telegram alert on every execution. Infrastructure: co-location server AWS eu-west-1. Monitoring dashboard: real-time P&L, latency metrics, success rate. All trades logged to PostgreSQL."},
    {"id": "N95", "domain": "Payments", "vagueness": "Detailed",
     "text": "Configure global payments for my B2B SaaS. Customers in US, EU, Japan, Brazil. Pricing in USD with local currency display. Tax: US state sales tax via TaxJar, EU VAT via API, JCT for Japan (10%), ISS for Brazil (2-5%). Payment methods: card (Stripe), SEPA direct debit (EU), Pix (Brazil), Konbini (Japan). Invoices in local language and currency. Stripe Billing + Avalara for tax. Daily sync to NetSuite ERP. Dunning for failed payments: 3 retries, customer email each attempt, suspend after 14 days."},
    {"id": "N96", "domain": "Payments", "vagueness": "Detailed",
     "text": "Implement a DeFi yield optimisation strategy with 100 ETH total capital. Allocation: 40 ETH → Lido staking (auto-compound stETH), 30 ETH → Aave supply as collateral, borrow 50% LTV in USDC, deploy USDC to Curve 3pool. 30 ETH reserve (no deployment). Rebalancing triggers: Aave LTV > 65% (deleverage), Lido APR drops > 2pp vs. 30-day avg (reduce allocation), Curve APY < 4% for 7 days (exit Curve). Daily report: yield breakdown, LTV, estimated APY. Emergency exit: full derisking within 4h if ETH drops >25% in 24h. Alerts via Telegram."},
    {"id": "N97", "domain": "Payments", "vagueness": "Detailed",
     "text": "I need an automated payroll system for my startup (25 employees, 8 countries: FR, UK, DE, US, SG, BR, IN, CA). Monthly pay cycle, processed on last local business day. Deductions: local pension contributions, income tax withholding, health insurance (country-specific rules). Employer tax filing automated where APIs exist. Integration: Rippling for HR headcount, Wise Business for international bank transfers. CFO approval gate for any batch > $50,000. Full audit trail, GDPR compliant. Employee payslips generated in local language."},
    {"id": "N98", "domain": "Payments", "vagueness": "Detailed",
     "text": "Configure a risk-managed crypto portfolio rebalancing system. Portfolio: $500k. Target allocation: BTC 35%, ETH 25%, SOL 10%, USDC 30%. Trigger rebalance if any asset drifts > 5% from target. Max single rebalance trade: $20,000. Slippage tolerance 0.2%. No trades during UTC 00:00–04:00. TWAP execution over 2 hours for trades > $10,000. Exchanges: Coinbase Pro and Kraken only (no Binance — compliance requirement). All rebalances logged to Airtable with timestamp, asset, quantity, price, and trigger reason. Monthly email report to investment committee."},
    {"id": "N99", "domain": "Payments", "vagueness": "Detailed",
     "text": "I'm building payment infrastructure for a Web3 gaming platform. Players buy in-game currency (GOLD): 1 USD = 100 GOLD, 1 ETH = 300,000 GOLD (live price + 5% spread). Players can cash out GOLD to ETH only (not USD), minimum 10,000 GOLD per withdrawal. AML controls: max $10,000/month per user, mandatory KYC (Onfido) above $1,000 lifetime spend. Smart contract on Polygon (upgradeable proxy). Backend: AWS. Regulatory framework: MiCA compliant (EU). Reserve fund: 100% of outstanding GOLD backed by ETH in multi-sig custody."},
    {"id": "N100", "domain": "Payments", "vagueness": "Detailed",
     "text": "Set up a complete trading risk management system for a family office. Account: $1M. Products: US equities and ETFs only (no options, no crypto, no leverage). Risk limits: max 5% in single stock, max 20% in single sector (GICS), max 2% daily drawdown (auto-halt + human override to resume), max 10 trades per day, no trading in first/last 10 minutes of session. Performance reporting: real-time P&L dashboard, weekly email to portfolio manager with attribution by sector, monthly report for trustee board. Custodian: Interactive Brokers via IBKR API. Compliance: all orders stored for SEC audit trail, 7-year retention policy."},
]


def load_existing_prompts() -> list:
    """Load the original 50 prompts from config."""
    with open(CONFIG_DIR / "prompts_P01-P50.json") as f:
        return json.load(f)


def load_existing_results() -> dict:
    """Load single-pass baseline scores already computed (if any)."""
    csv_path = OUTPUT_DIR / "baseline_extended_n150.csv"
    if not csv_path.exists():
        return {}
    done = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done[row["prompt_id"]] = float(row["final_cov"])
    return done


def score_single_pass(prompt_data: dict, gemini_client) -> dict:
    """Score a prompt with Gemini 2.5 Flash single-pass (§8 protocol)."""
    domain = prompt_data["domain"]
    full_prompt = f"{SINGLE_PASS_SYSTEM}\n\nUser request:\n{prompt_data['text']}"

    try:
        response_text = call_gemini(full_prompt, client=gemini_client,
                                    temperature=0.0, max_tokens=3000)
        time.sleep(5)  # Rate limit: free tier
    except Exception as e:
        print(f"    [ERROR] Gemini call failed: {e}")
        time.sleep(15)
        return None

    try:
        time.sleep(4)
        score = score_response(response_text, domain, prompt_data["text"],
                               gemini_client=gemini_client, scorer="gemini")
        return score
    except Exception as e:
        print(f"    [SCORING ERROR] {e}")
        try:
            score = score_response(response_text, domain, prompt_data["text"],
                                   scorer="ollama")
            return score
        except Exception:
            return None


def bootstrap_ci(scores: list, n_bootstrap: int = 10000, alpha: float = 0.05) -> tuple:
    """Return (mean, std, ci_lower, ci_upper) with bootstrap 95% CI."""
    arr = np.array(scores)
    rng = np.random.default_rng(SEED)
    boot_means = [np.mean(rng.choice(arr, len(arr), replace=True))
                  for _ in range(n_bootstrap)]
    ci_lo = np.percentile(boot_means, 100 * alpha / 2)
    ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(np.std(arr)), float(ci_lo), float(ci_hi)


def compute_stats(results: list) -> dict:
    """Compute mean/std/CI for all groupings."""
    stats = {"n_total": len(results), "generated_at": datetime.now(timezone.utc).isoformat()}

    all_covs = [r["final_cov"] for r in results if r["final_cov"] is not None]
    mean, std, ci_lo, ci_hi = bootstrap_ci(all_covs)
    stats["overall"] = {"n": len(all_covs), "mean": round(mean, 4), "std": round(std, 4),
                        "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4),
                        "median": round(float(np.median(all_covs)), 4)}

    stats["by_domain"] = {}
    for domain in ["Consulting", "Medical", "Payments"]:
        covs = [r["final_cov"] for r in results if r["domain"] == domain and r["final_cov"] is not None]
        if covs:
            mean, std, ci_lo, ci_hi = bootstrap_ci(covs)
            stats["by_domain"][domain] = {
                "n": len(covs), "mean": round(mean, 4), "std": round(std, 4),
                "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4)
            }

    stats["by_vagueness"] = {}
    for vag in ["Ultra-vague", "Medium", "Detailed"]:
        covs = [r["final_cov"] for r in results if r["vagueness"] == vag and r["final_cov"] is not None]
        if covs:
            mean, std, ci_lo, ci_hi = bootstrap_ci(covs)
            stats["by_vagueness"][vag] = {
                "n": len(covs), "mean": round(mean, 4), "std": round(std, 4),
                "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4)
            }

    stats["by_domain_vagueness"] = {}
    for domain in ["Consulting", "Medical", "Payments"]:
        stats["by_domain_vagueness"][domain] = {}
        for vag in ["Ultra-vague", "Medium", "Detailed"]:
            covs = [r["final_cov"] for r in results
                    if r["domain"] == domain and r["vagueness"] == vag and r["final_cov"] is not None]
            if covs:
                mean, std, ci_lo, ci_hi = bootstrap_ci(covs)
                stats["by_domain_vagueness"][domain][vag] = {
                    "n": len(covs), "mean": round(mean, 4), "std": round(std, 4),
                    "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4)
                }

    return stats


def generate_latex_table(stats: dict, results: list, output_path: Path):
    """Generate updated §8 LaTeX table."""
    n50_covs = [r["final_cov"] for r in results
                if r["prompt_id"].startswith("P") and r["final_cov"] is not None]
    n100_covs = [r["final_cov"] for r in results
                 if r["prompt_id"].startswith("N") and r["final_cov"] is not None]

    n50_mean = np.mean(n50_covs) * 100 if n50_covs else 0
    n50_std = np.std(n50_covs) * 100 if n50_covs else 0
    n100_mean = np.mean(n100_covs) * 100 if n100_covs else 0

    o = stats["overall"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Baseline coverage (Gemini 2.5 Flash, single-pass) on extended dataset "
        r"$n=150$ (50 prompts per domain). 95\% bootstrap CI reported. "
        r"Original $n=50$ included for comparison.}",
        r"\label{tab:baseline_extended}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Group} & \textbf{n} & \textbf{Mean cov.} & \textbf{Std} "
        r"& \textbf{95\% CI} \\",
        r"\midrule",
        f"  Original set ($n=50$) & 50 & {n50_mean:.1f}\\% & {n50_std:.1f}\\% & -- \\\\",
        r"\midrule",
    ]

    for domain in ["Consulting", "Medical", "Payments"]:
        d = stats["by_domain"].get(domain, {})
        lines.append(
            f"  {domain} & {d.get('n','--')} & {d.get('mean',0)*100:.1f}\\% "
            f"& {d.get('std',0)*100:.1f}\\% "
            f"& [{d.get('ci_95_lo',0)*100:.1f}, {d.get('ci_95_hi',0)*100:.1f}]\\% \\\\"
        )

    lines.append(r"\midrule")
    for vag in ["Ultra-vague", "Medium", "Detailed"]:
        v = stats["by_vagueness"].get(vag, {})
        lines.append(
            f"  \\textit{{{vag}}} & {v.get('n','--')} & {v.get('mean',0)*100:.1f}\\% "
            f"& {v.get('std',0)*100:.1f}\\% "
            f"& [{v.get('ci_95_lo',0)*100:.1f}, {v.get('ci_95_hi',0)*100:.1f}]\\% \\\\"
        )

    lines.extend([
        r"\midrule",
        f"  \\textbf{{Overall ($n=150$)}} & {o['n']} & \\textbf{{{o['mean']*100:.1f}\\%}} "
        f"& {o['std']*100:.1f}\\% "
        f"& [{o['ci_95_lo']*100:.1f}, {o['ci_95_hi']*100:.1f}]\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  → LaTeX: {output_path}")


def save_results(results: list):
    """Save all scored results to CSV."""
    csv_path = OUTPUT_DIR / "baseline_extended_n150.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prompt_id", "domain", "vagueness", "final_cov",
            "n_resolved", "n_mentioned", "n_absent", "set"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    print(f"  → CSV: {csv_path} ({len(results)} rows)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"EXPERIMENT 2: Baseline Extension n=150")
    print(f"Original: 50 prompts | New: {len(NEW_PROMPTS)} prompts | Total: 150")
    print(f"Scorer: Gemini 2.5 Flash single-pass")
    print(f"{'='*60}\n")

    gemini_client = get_gemini_client()
    all_prompts = load_existing_prompts() + NEW_PROMPTS
    assert len(all_prompts) == 150, f"Expected 150 prompts, got {len(all_prompts)}"

    # Validate distribution
    for domain in ["Consulting", "Medical", "Payments"]:
        domain_ps = [p for p in all_prompts if p["domain"] == domain]
        print(f"  {domain}: {len(domain_ps)} prompts total")
        for vag in ["Ultra-vague", "Medium", "Detailed"]:
            n = sum(1 for p in domain_ps if p["vagueness"] == vag)
            print(f"    {vag}: {n}")

    # Load existing scores (resume)
    existing_scores = {} if args.no_resume else load_existing_results()
    print(f"\n  Resume: {len(existing_scores)} prompts already scored.")

    results = []
    pending = [p for p in all_prompts if p["id"] not in existing_scores]
    print(f"  Pending: {len(pending)} prompts to score.\n")

    for i, p in enumerate(pending):
        tag = "orig" if p["id"].startswith("P") else "new"
        total_done = len(existing_scores) + i + 1
        print(f"  [{total_done}/150] {p['id']} [{tag}] | {p['domain']} | {p['vagueness']}")
        print(f"    \"{p['text'][:70]}{'...' if len(p['text'])>70 else ''}\"")

        score = score_single_pass(p, gemini_client)
        if score is None:
            print(f"    [SKIP] Scoring failed, recording 0.0")
            cov, n_res, n_men, n_abs = 0.0, 0, 0, 0
        else:
            cov = score["coverage"]
            n_res = len(score.get("params_resolved", []))
            n_men = len(score.get("params_mentioned", []))
            n_abs = len(score.get("params_absent", []))

        existing_scores[p["id"]] = cov
        results.append({
            "prompt_id": p["id"], "domain": p["domain"], "vagueness": p["vagueness"],
            "final_cov": cov, "n_resolved": n_res, "n_mentioned": n_men,
            "n_absent": n_abs, "set": tag
        })
        print(f"    → cov={cov:.1%} | resolved={n_res} | mentioned={n_men} | absent={n_abs}")

        # Save incrementally
        if (i + 1) % 5 == 0 or i == len(pending) - 1:
            # Merge existing + new results
            all_results = list(results)
            for pid, cov_val in existing_scores.items():
                if not any(r["prompt_id"] == pid for r in all_results):
                    prompt = next(p2 for p2 in all_prompts if p2["id"] == pid)
                    all_results.append({
                        "prompt_id": pid, "domain": prompt["domain"],
                        "vagueness": prompt["vagueness"], "final_cov": cov_val,
                        "n_resolved": "", "n_mentioned": "", "n_absent": "",
                        "set": "orig" if pid.startswith("P") else "new"
                    })
            save_results(all_results)

        time.sleep(2)

    # Final: merge everything
    final_results = []
    for p in all_prompts:
        cov_val = existing_scores.get(p["id"])
        match = next((r for r in results if r["prompt_id"] == p["id"]), None)
        if match:
            final_results.append(match)
        elif cov_val is not None:
            final_results.append({
                "prompt_id": p["id"], "domain": p["domain"], "vagueness": p["vagueness"],
                "final_cov": cov_val, "n_resolved": "", "n_mentioned": "", "n_absent": "",
                "set": "orig" if p["id"].startswith("P") else "new"
            })

    save_results(final_results)

    # Save new prompts CSV
    new_csv = OUTPUT_DIR / "prompts_new_100.csv"
    with open(new_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "domain", "vagueness", "text"])
        writer.writeheader()
        for p in NEW_PROMPTS:
            writer.writerow(p)
    print(f"  → New prompts: {new_csv}")

    # Stats
    valid = [r for r in final_results if r["final_cov"] != "" and r["final_cov"] is not None]
    stats = compute_stats(valid)

    stats_path = OUTPUT_DIR / "baseline_extended_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  → Stats: {stats_path}")

    generate_latex_table(stats, valid, OUTPUT_DIR / "table_baseline_latex.tex")

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS — BASELINE EXTENDED n=150")
    print(f"{'='*60}")
    o = stats["overall"]
    print(f"\nOverall: {o['mean']*100:.1f}% ± {o['std']*100:.1f}% "
          f"[95% CI: {o['ci_95_lo']*100:.1f}%–{o['ci_95_hi']*100:.1f}%]")
    for domain, d in stats["by_domain"].items():
        print(f"  {domain}: {d['mean']*100:.1f}% ± {d['std']*100:.1f}% "
              f"[{d['ci_95_lo']*100:.1f}–{d['ci_95_hi']*100:.1f}%]")
    print()
    for vag, v in stats["by_vagueness"].items():
        print(f"  {vag}: {v['mean']*100:.1f}% ± {v['std']*100:.1f}% "
              f"[{v['ci_95_lo']*100:.1f}–{v['ci_95_hi']*100:.1f}%]")
    print(f"\nDone. All results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
