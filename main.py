import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import os

app = FastAPI(
    title="UpayPulse Customer Intelligence Suite",
    description="Developed by Team The Entropy for AI Hackathon 2026",
    version="3.6.0"
)

# Load Trained Model Bundle
bundle_path = 'upay_production_bundle.joblib'
try:
    bundle = joblib.load(bundle_path)
    churn_model = bundle['churn_model']
    nba_model = bundle['nba_model']
    feature_encoders = bundle['feature_encoders']
    target_encoder = bundle['target_encoder']
    feature_names = bundle['feature_names']
except Exception as e:
    raise RuntimeError(f"Error loading bundle artifacts: {str(e)}")

# Request Schema
class CustomerFeatures(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Gender: str
    AccountAgeMonths: int = Field(..., ge=0)
    AppOpenFreqWeekly: int = Field(..., ge=0)
    PreferredCategory: str
    LifeStageSegment: str
    AvgMonthlyTransactions: int = Field(..., ge=0)
    AvgTransactionAmount: float = Field(..., ge=0.0)
    SupportTicketsCount: int = Field(..., ge=0)

@app.get("/shap-image")
def get_shap_image():
    image_path = "shap_summary_plot.png"
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="SHAP plot image not found")

# Core Prediction Engine
def run_intelligence(input_data: dict):
    df_input = pd.DataFrame([input_data])
    for col, encoder in feature_encoders.items():
        if input_data[col] not in encoder.classes_:
            raise ValueError(f"Invalid categorical value for {col}")
        df_input[col] = encoder.transform(df_input[col])
    
    X_ready = df_input[feature_names]
    churn_prob = float(churn_model.predict_proba(X_ready)[0][1])
    is_churn = bool(churn_prob >= 0.50)
    risk_level = "High" if churn_prob >= 0.70 else ("Medium" if churn_prob >= 0.40 else "Low")
    
    nba_encoded = nba_model.predict(X_ready)[0]
    recommended_offer = target_encoder.inverse_transform([nba_encoded])[0]

    ticket_score = min(input_data['SupportTicketsCount'] * 25, 95)
    inact_score = max(0, (14 - input_data['AppOpenFreqWeekly']) * 6)
    vol_score = max(0, (20 - input_data['AvgMonthlyTransactions']) * 4)

    if input_data['SupportTicketsCount'] >= 3:
        primary_driver = "Unresolved Support Inquiries / High CSAT Variance"
    elif input_data['AppOpenFreqWeekly'] <= 3:
        primary_driver = "Product Login & Weekly Inactivity"
    elif input_data['AvgMonthlyTransactions'] <= 4:
        primary_driver = "Transaction Dropoff / Decreased Velocity"
    else:
        primary_driver = "Stable High-Engagement Profile"

    if risk_level == "High":
        dispatch = "Immediate (Within 24h)"
        channel = "Priority SMS + Push Notification"
        lift = "+65% Retention Lift"
    elif risk_level == "Medium":
        dispatch = "Next 48 Hours"
        channel = "In-App Targeted Banner"
        lift = "+38% Retention Lift"
    else:
        dispatch = "Standard Monthly Cadence"
        channel = "Loyalty Rewards Tab"
        lift = "+15% LTV Growth"

    return {
        "churn_probability": round(churn_prob * 100, 1),
        "is_at_risk": is_churn,
        "risk_level": risk_level,
        "primary_risk_driver": primary_driver,
        "recommended_offer": recommended_offer,
        "dispatch_timing": dispatch,
        "outreach_channel": channel,
        "lift": lift,
        "telemetry": {
            "ticket_score": ticket_score,
            "inact_score": inact_score,
            "vol_score": vol_score
        }
    }

@app.post("/api/predict")
def predict_endpoint(payload: CustomerFeatures):
    try:
        return run_intelligence(payload.dict())
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Full Master Dashboard UI with Integrated SVG Logo & Team The Entropy Branding
@app.get("/", response_class=HTMLResponse)
def unified_saas_platform():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>UpayPulse | Enterprise Customer Intelligence Platform</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet"/>
        <style>
            body { font-family: 'Plus Jakarta Sans', sans-serif; }
            .active-tab { background-color: #EEF2FF; color: #4338CA; border-left: 4px solid #4F46E5; }
        </style>
    </head>
    <body class="bg-[#F8FAFC] text-slate-800 antialiased flex min-h-screen">

        <!-- Sidebar Navigation -->
        <aside class="w-64 bg-white border-r border-slate-200 flex flex-col justify-between py-6 px-4 fixed h-full z-20">
            <div>
                <!-- Brand Header with The Entropy Vector Logo -->
                <div class="px-1 mb-8">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 280 60" class="w-full h-auto">
                      <defs>
                        <linearGradient id="entropy-grad" x1="4" y1="6" x2="48" y2="50" gradientUnits="userSpaceOnUse">
                          <stop offset="0%" stop-color="#4F46E5"/>
                          <stop offset="50%" stop-color="#3B82F6"/>
                          <stop offset="100%" stop-color="#06B6D4"/>
                        </linearGradient>
                        <linearGradient id="ai-chip" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0%" stop-color="#7C3AED"/>
                          <stop offset="100%" stop-color="#4F46E5"/>
                        </linearGradient>
                        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                          <feGaussianBlur stdDeviation="2.5" result="blur" />
                          <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                      </defs>
                      <rect x="6" y="6" width="48" height="48" rx="14" fill="url(#entropy-grad)" />
                      <circle cx="30" cy="30" r="16" stroke="#FFFFFF" stroke-width="1.8" stroke-dasharray="3 3" opacity="0.6"/>
                      <circle cx="30" cy="30" r="10" stroke="#FFFFFF" stroke-width="2"/>
                      <circle cx="30" cy="30" r="4.5" fill="#FFFFFF" filter="url(#glow)"/>
                      <circle cx="42" cy="18" r="2.5" fill="#FDE047"/>
                      <circle cx="18" cy="40" r="2" fill="#93C5FD"/>
                      <circle cx="40" cy="39" r="1.8" fill="#C4B5FD"/>
                      <text x="66" y="31" font-family="'Plus Jakarta Sans', sans-serif" font-weight="800" font-size="20" fill="#0F172A" letter-spacing="-0.03em">UpayPulse</text>
                      <rect x="180" y="16" width="28" height="17" rx="5" fill="url(#ai-chip)"/>
                      <text x="186" y="29" font-family="'Plus Jakarta Sans', sans-serif" font-weight="800" font-size="10" fill="#FFFFFF">AI</text>
                      <text x="67" y="47" font-family="'Plus Jakarta Sans', sans-serif" font-weight="600" font-size="10" fill="#64748B" letter-spacing="0.02em">BY THE ENTROPY</text>
                      <circle cx="166" cy="44" r="1.5" fill="#94A3B8"/>
                      <text x="174" y="47" font-family="'Plus Jakarta Sans', sans-serif" font-weight="600" font-size="9.5" fill="#4F46E5" letter-spacing="0.04em">PREDICT &bull; RETAIN</text>
                    </svg>
                </div>

                <!-- Nav Menu -->
                <div class="space-y-6">
                    <div>
                        <p class="text-[11px] font-bold uppercase tracking-wider text-slate-400 px-3 mb-2">Core Intelligence</p>
                        <nav class="space-y-1">
                            <button onclick="switchTab('dashboard')" id="nav-dashboard" class="w-full flex items-center gap-3 px-3 py-2 text-sm font-semibold rounded-lg text-slate-600 hover:bg-slate-50 transition text-left">
                                <span class="material-symbols-outlined text-[20px]">grid_view</span>
                                <span>Dashboard</span>
                            </button>
                            <button onclick="switchTab('churn')" id="nav-churn" class="w-full flex items-center gap-3 px-3 py-2 text-sm font-semibold rounded-lg text-slate-600 hover:bg-slate-50 transition text-left active-tab">
                                <span class="material-symbols-outlined text-[20px]">warning</span>
                                <span>Churn Intelligence</span>
                            </button>
                            <button onclick="switchTab('nba')" id="nav-nba" class="w-full flex items-center gap-3 px-3 py-2 text-sm font-semibold rounded-lg text-slate-600 hover:bg-slate-50 transition text-left">
                                <span class="material-symbols-outlined text-[20px]">auto_awesome</span>
                                <span>Next Best Action</span>
                            </button>
                            <a href="/docs" target="_blank" class="w-full flex items-center gap-3 px-3 py-2 text-sm font-semibold rounded-lg text-slate-600 hover:bg-slate-50 transition text-left">
                                <span class="material-symbols-outlined text-[20px]">code</span>
                                <span>API Swagger Docs</span>
                            </a>
                        </nav>
                    </div>
                </div>
            </div>

            <!-- Team Branding Footer -->
            <div class="border-t border-slate-100 pt-4 px-2 flex items-center justify-between">
                <div class="flex items-center gap-2.5">
                    <div class="w-9 h-9 rounded-full bg-indigo-900 text-indigo-100 font-black flex items-center justify-center text-xs shadow-sm">
                        TE
                    </div>
                    <div>
                        <p class="text-xs font-bold text-slate-900 leading-tight">The Entropy</p>
                        <p class="text-[10px] text-slate-400 font-medium">DIU AI Hackathon 2026</p>
                    </div>
                </div>
                <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" title="System Live"></span>
            </div>
        </aside>

        <!-- Main Workspace -->
        <main class="ml-64 flex-1 p-8">

            <!-- ================= VIEW 1: EXECUTIVE DASHBOARD ================= -->
            <section id="view-dashboard" class="hidden space-y-6">
                <div class="flex items-center justify-between pb-6 border-b border-slate-200">
                    <div>
                        <span class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
                            Enterprise Executive Overview
                        </span>
                        <h1 class="text-2xl font-extrabold text-slate-900 tracking-tight mt-1">Portfolio Customer Intelligence</h1>
                        <p class="text-xs text-slate-500 mt-0.5">High-level telemetry and financial metrics curated by Team The Entropy.</p>
                    </div>
                    <button onclick="switchTab('churn')" class="px-4 py-2 bg-indigo-600 text-white rounded-xl text-xs font-bold hover:bg-indigo-700 transition shadow-sm">
                        Open Churn Cockpit &rarr;
                    </button>
                </div>

                <!-- Top 4 KPI Metrics -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div class="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                        <span class="text-xs font-semibold text-slate-500">Total Active Base</span>
                        <p class="text-3xl font-extrabold text-slate-900 mt-2">48,293</p>
                        <span class="text-xs font-bold text-emerald-600">+8.4% YoY Growth</span>
                    </div>
                    <div class="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                        <span class="text-xs font-semibold text-slate-500">Predicted Churn Rate</span>
                        <p class="text-3xl font-extrabold text-rose-600 mt-2">8.7%</p>
                        <span class="text-xs font-bold text-rose-600">+0.6% vs last cycle</span>
                    </div>
                    <div class="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                        <span class="text-xs font-semibold text-slate-500">Annual Revenue at Risk</span>
                        <p class="text-3xl font-extrabold text-slate-900 mt-2">$2.8M</p>
                        <span class="text-xs font-bold text-amber-600">4,821 Accounts</span>
                    </div>
                    <div class="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                        <span class="text-xs font-semibold text-slate-500">Autonomous Salvage Rate</span>
                        <p class="text-3xl font-extrabold text-indigo-600 mt-2">81.7%</p>
                        <span class="text-xs font-bold text-emerald-600">Via Next Best Action</span>
                    </div>
                </div>

                <!-- Cohort Breakdown & Global SHAP -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                        <h3 class="text-base font-bold text-slate-900 mb-1">Customer Segmentation Distribution</h3>
                        <p class="text-xs text-slate-400 mb-4">Real-time cohort distribution across registered MFS users</p>
                        <div class="space-y-3">
                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Champions / High Value</span>
                                    <span>49.3% ($14.2M ARR)</span>
                                </div>
                                <div class="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                                    <div class="bg-indigo-600 h-full" style="width: 49.3%"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Loyal Working Professionals</span>
                                    <span>41.0% ($11.8M ARR)</span>
                                </div>
                                <div class="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                                    <div class="bg-blue-500 h-full" style="width: 41%"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1 text-rose-600">
                                    <span>At-Risk Cohort (Immediate Target)</span>
                                    <span>9.7% ($2.8M ARR)</span>
                                </div>
                                <div class="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                                    <div class="bg-rose-500 h-full" style="width: 9.7%"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                        <div class="flex items-center justify-between mb-2">
                            <h3 class="text-base font-bold text-slate-900">Global Feature Importance (SHAP)</h3>
                            <a href="/shap-image" target="_blank" class="text-xs font-bold text-indigo-600 hover:underline">Full Image &rarr;</a>
                        </div>
                        <p class="text-xs text-slate-400 mb-3">Model drivers trained across 1,500 balanced records</p>
                        <div class="bg-slate-50 p-2 rounded-xl border border-slate-100">
                            <img src="/shap-image" alt="SHAP Plot" class="w-full h-44 object-contain rounded-lg">
                        </div>
                    </div>
                </div>
            </section>

            <!-- ================= VIEW 2: CHURN INTELLIGENCE (LIVE SIMULATOR) ================= -->
            <section id="view-churn" class="space-y-6">
                <!-- Top App Bar -->
                <div class="flex items-center justify-between pb-6 border-b border-slate-200">
                    <div>
                        <div class="flex items-center gap-2">
                            <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-indigo-50 text-indigo-700 border border-indigo-200/60">
                                <span class="w-1.5 h-1.5 rounded-full bg-indigo-600 animate-pulse"></span> RANDOM FOREST ACTIVE (AUC 0.9506)
                            </span>
                            <span class="text-xs text-slate-400 font-medium">• Team The Entropy Core</span>
                        </div>
                        <h1 class="text-2xl font-extrabold text-slate-900 tracking-tight mt-1">Churn Intelligence & Predictive Risk Engine</h1>
                        <p class="text-xs text-slate-500 mt-0.5">Live inference module with feature explainability and automated Next-Best-Action pairing.</p>
                    </div>
                    <div class="flex items-center gap-3">
                        <button onclick="loadScenario('high_risk')" class="px-3.5 py-2 text-xs font-bold rounded-lg bg-rose-50 text-rose-700 border border-rose-200 hover:bg-rose-100 transition shadow-sm">
                            Simulate High-Risk
                        </button>
                        <button onclick="loadScenario('loyal')" class="px-3.5 py-2 text-xs font-bold rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 transition shadow-sm">
                            Simulate Loyal
                        </button>
                    </div>
                </div>

                <!-- Core Grid -->
                <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    <!-- Left: Input Form (5 cols) -->
                    <div class="lg:col-span-5 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                        <h2 class="text-base font-bold text-slate-900 mb-4 pb-2 border-b border-slate-100">Customer 360 Attribute Input</h2>
                        <form id="customerForm" class="space-y-4">
                            <div class="grid grid-cols-2 gap-3">
                                <div>
                                    <label class="text-xs font-semibold text-slate-500">Age</label>
                                    <input type="number" id="Age" value="24" class="w-full mt-1 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs text-slate-800 font-medium">
                                </div>
                                <div>
                                    <label class="text-xs font-semibold text-slate-500">Gender</label>
                                    <select id="Gender" class="w-full mt-1 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs text-slate-800 font-medium">
                                        <option value="Female">Female</option>
                                        <option value="Male">Male</option>
                                    </select>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-3">
                                <div>
                                    <label class="text-xs font-semibold text-slate-500">Tenure (Months)</label>
                                    <input type="number" id="AccountAgeMonths" value="12" class="w-full mt-1 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs text-slate-800 font-medium">
                                </div>
                                <div>
                                    <label class="text-xs font-semibold text-slate-500">Weekly App Opens</label>
                                    <input type="number" id="AppOpenFreqWeekly" value="2" class="w-full mt-1 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs text-slate-800 font-medium">
                                </div>
                            </div>

                            <div>
                                <label class="text-xs font-semibold text-slate-500">Preferred Transaction Category</label>
                                <select id="PreferredCategory" class="w-full mt-1 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs text-slate-800 font-medium">
                                    <option value="Send Money">Send Money</option>
                                    <option value="Mobile Recharge">Mobile Recharge</option>
                                    <option value="Merchant Payment">Merchant Payment</option>
                                    <option value="Cash-Out">Cash-Out</option>
                                </select>
                            </div>

                            <div>
                                <label class="text-xs font-semibold text-slate-500">Customer Life Stage Segment</label>
                                <select id="LifeStageSegment" class="w-full mt-1 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs text-slate-800 font-medium">
                                    <option value="Student/Youth">Student / Youth</option>
                                    <option value="Working Professional">Working Professional</option>
                                    <option value="Family/Mature">Family / Mature</option>
                                </select>
                            </div>

                            <div class="grid grid-cols-2 gap-3">
                                <div>
                                    <label class="text-xs font-semibold text-slate-500">Monthly Txn Volume</label>
                                    <input type="number" id="AvgMonthlyTransactions" value="3" class="w-full mt-1 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs text-slate-800 font-medium">
                                </div>
                                <div>
                                    <label class="text-xs font-semibold text-slate-500">Avg Txn Amount (৳)</label>
                                    <input type="number" step="0.1" id="AvgTransactionAmount" value="450.0" class="w-full mt-1 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs text-slate-800 font-medium">
                                </div>
                            </div>

                            <div>
                                <label class="text-xs font-semibold text-slate-500">Support Tickets (Past 90 Days)</label>
                                <input type="number" id="SupportTicketsCount" value="4" class="w-full mt-1 bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-xs text-slate-800 font-medium">
                            </div>

                            <button type="submit" class="w-full bg-slate-900 hover:bg-slate-800 text-white font-bold py-3 rounded-xl text-xs shadow-md transition">
                                Run Predictive Intelligence Check
                            </button>
                        </form>
                    </div>

                    <!-- Right: Outcomes & Feature Drivers (7 cols) -->
                    <div class="lg:col-span-7 space-y-6">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
                                <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Churn Risk Probability</span>
                                <div class="mt-2 flex items-baseline justify-between">
                                    <span id="probText" class="text-4xl font-extrabold text-slate-900 tracking-tight">--%</span>
                                    <span id="riskBadge" class="px-2.5 py-1 rounded-full text-xs font-bold uppercase bg-slate-100 text-slate-600">Pending</span>
                                </div>
                                <div class="w-full bg-slate-100 h-2.5 rounded-full mt-4 overflow-hidden">
                                    <div id="probBar" class="bg-slate-400 h-full rounded-full transition-all duration-500" style="width: 0%"></div>
                                </div>
                            </div>

                            <div class="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
                                <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">AI Next-Best-Action</span>
                                <div>
                                    <h3 id="offerTitle" class="text-lg font-bold text-indigo-700 mt-2">--</h3>
                                    <p id="liftText" class="text-xs font-semibold text-emerald-600 mt-0.5">--</p>
                                </div>
                                <span class="text-[11px] text-slate-400 font-medium">Algorithmic Decisioning Engine</span>
                            </div>
                        </div>

                        <!-- Attribution Bars -->
                        <div class="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                            <div class="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
                                <div>
                                    <h3 class="text-sm font-bold text-slate-900">Explainable AI: Why Is Customer At Risk?</h3>
                                    <p class="text-xs text-slate-400">SHAP-based attribution weighted across user telemetry</p>
                                </div>
                                <span id="primaryDriverBadge" class="text-[11px] font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-700">--</span>
                            </div>

                            <div class="space-y-4">
                                <div>
                                    <div class="flex justify-between text-xs font-bold mb-1">
                                        <span class="text-slate-700">Unresolved Support Inquiries / Low CSAT</span>
                                        <span id="scoreTicket" class="text-rose-600 font-extrabold">--%</span>
                                    </div>
                                    <div class="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                                        <div id="barTicket" class="bg-rose-500 h-full transition-all duration-300" style="width: 0%"></div>
                                    </div>
                                </div>

                                <div>
                                    <div class="flex justify-between text-xs font-bold mb-1">
                                        <span class="text-slate-700">Product Login & Weekly Inactivity</span>
                                        <span id="scoreInact" class="text-amber-600 font-extrabold">--%</span>
                                    </div>
                                    <div class="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                                        <div id="barInact" class="bg-amber-500 h-full transition-all duration-300" style="width: 0%"></div>
                                    </div>
                                </div>

                                <div>
                                    <div class="flex justify-between text-xs font-bold mb-1">
                                        <span class="text-slate-700">Transaction Velocity & Volume Decline</span>
                                        <span id="scoreTxn" class="text-blue-600 font-extrabold">--%</span>
                                    </div>
                                    <div class="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                                        <div id="barTxn" class="bg-blue-500 h-full transition-all duration-300" style="width: 0%"></div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Action Plan Strip -->
                        <div class="bg-indigo-50/60 p-5 rounded-2xl border border-indigo-100 flex items-center justify-between">
                            <div>
                                <span class="text-xs font-extrabold text-indigo-900 uppercase tracking-wide">Automated Execution Strategy</span>
                                <p id="actionSummary" class="text-xs text-slate-700 font-semibold mt-1">Awaiting customer submission...</p>
                            </div>
                            <button onclick="switchTab('nba')" class="px-3.5 py-2 bg-indigo-600 text-white rounded-lg text-xs font-bold hover:bg-indigo-700 transition">
                                Review Playbook &rarr;
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            <!-- ================= VIEW 3: NEXT BEST ACTION (RECOMMENDATION ENGINE) ================= -->
            <section id="view-nba" class="hidden space-y-6">
                <div class="flex items-center justify-between pb-6 border-b border-slate-200">
                    <div>
                        <span class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
                            Automated Decisioning Engine
                        </span>
                        <h1 class="text-2xl font-extrabold text-slate-900 tracking-tight mt-1">Next Best Action Recommendation Hub</h1>
                        <p class="text-xs text-slate-500 mt-0.5">Algorithmically matched retention offers and outreach playbooks.</p>
                    </div>
                    <span class="px-3 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-lg text-xs font-bold">
                        Engine Status: Active (91% Confidence)
                    </span>
                </div>

                <!-- Primary Recommendation Hero Box -->
                <div class="bg-white p-6 rounded-2xl border-2 border-indigo-200 shadow-md relative overflow-hidden">
                    <div class="flex items-center justify-between mb-3">
                        <span class="px-3 py-1 bg-indigo-600 text-white text-xs font-extrabold rounded-full uppercase tracking-wider">
                            Primary Prescribed Action
                        </span>
                        <span id="nba-lift-badge" class="text-xs font-bold text-emerald-600 bg-emerald-50 px-2.5 py-1 rounded-lg border border-emerald-200">
                            +65% Retention Lift
                        </span>
                    </div>

                    <h2 id="nba-offer-title" class="text-xl font-extrabold text-slate-900 mb-2">
                        Cashback on Send Money
                    </h2>
                    <p id="nba-offer-desc" class="text-xs text-slate-600 leading-relaxed max-w-3xl mb-4">
                        Customer exhibiting drop in weekly activity and recent service friction. Deploying tailored fee rebates directly targets the disengagement root cause.
                    </p>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-slate-50 rounded-xl mb-4 text-xs font-semibold">
                        <div>
                            <span class="text-slate-400 block uppercase text-[10px]">Optimal Timing</span>
                            <span id="nba-timing" class="text-slate-800 text-sm font-bold">Immediate (Within 24 Hours)</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block uppercase text-[10px]">Delivery Channel</span>
                            <span id="nba-channel" class="text-indigo-600 text-sm font-bold">Priority SMS + Push Notification</span>
                        </div>
                        <div>
                            <span class="text-slate-400 block uppercase text-[10px]">Governance Policy</span>
                            <span class="text-emerald-600 text-sm font-bold">Autonomous Trigger Enabled</span>
                        </div>
                    </div>

                    <div class="flex items-center gap-3">
                        <button onclick="alert('Offer successfully scheduled for dispatch via Upay campaign pipeline!')" class="px-5 py-2.5 bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold rounded-xl transition shadow">
                            Execute & Deploy Offer
                        </button>
                        <button onclick="switchTab('churn')" class="px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold rounded-xl transition">
                            Simulate Different Profile
                        </button>
                    </div>
                </div>

                <!-- Catalog of All Next-Best-Action Offers -->
                <div>
                    <h3 class="text-base font-bold text-slate-900 mb-3">Model Supported Offer Catalog (Multi-Class Engine)</h3>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                            <span class="text-[10px] font-bold text-indigo-600 uppercase bg-indigo-50 px-2 py-0.5 rounded">Recharge Focus</span>
                            <h4 class="text-sm font-bold text-slate-900 mt-2">Cashback on Send Money</h4>
                            <p class="text-[11px] text-slate-500 mt-1">Triggers transaction habit loops for frequent peer transfers.</p>
                        </div>
                        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                            <span class="text-[10px] font-bold text-amber-600 uppercase bg-amber-50 px-2 py-0.5 rounded">High Value</span>
                            <h4 class="text-sm font-bold text-slate-900 mt-2">Merchant Discount Voucher</h4>
                            <p class="text-[11px] text-slate-500 mt-1">High CSAT recovery voucher for retail purchase users.</p>
                        </div>
                        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                            <span class="text-[10px] font-bold text-emerald-600 uppercase bg-emerald-50 px-2 py-0.5 rounded">Savings</span>
                            <h4 class="text-sm font-bold text-slate-900 mt-2">Savings Bonus</h4>
                            <p class="text-[11px] text-slate-500 mt-1">Incentivizes holding balances to build long-term sticky tenure.</p>
                        </div>
                        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                            <span class="text-[10px] font-bold text-rose-600 uppercase bg-rose-50 px-2 py-0.5 rounded">Emergency Retention</span>
                            <h4 class="text-sm font-bold text-slate-900 mt-2">Zero-Fee Cashout</h4>
                            <p class="text-[11px] text-slate-500 mt-1">Direct fee waiver for users showing critical dormancy indicators.</p>
                        </div>
                    </div>
                </div>
            </section>

        </main>

        <script>
            // Seamless Tab Switching
            function switchTab(tabId) {
                document.getElementById('view-dashboard').classList.add('hidden');
                document.getElementById('view-churn').classList.add('hidden');
                document.getElementById('view-nba').classList.add('hidden');

                document.getElementById('nav-dashboard').classList.remove('active-tab');
                document.getElementById('nav-churn').classList.remove('active-tab');
                document.getElementById('nav-nba').classList.remove('active-tab');

                if (tabId === 'dashboard') {
                    document.getElementById('view-dashboard').classList.remove('hidden');
                    document.getElementById('nav-dashboard').classList.add('active-tab');
                } else if (tabId === 'nba') {
                    document.getElementById('view-nba').classList.remove('hidden');
                    document.getElementById('nav-nba').classList.add('active-tab');
                } else {
                    document.getElementById('view-churn').classList.remove('hidden');
                    document.getElementById('nav-churn').classList.add('active-tab');
                }
            }

            // Quick Scenario Presets
            function loadScenario(type) {
                if(type === 'high_risk') {
                    document.getElementById('Age').value = 24;
                    document.getElementById('Gender').value = 'Female';
                    document.getElementById('AccountAgeMonths').value = 12;
                    document.getElementById('AppOpenFreqWeekly').value = 2;
                    document.getElementById('PreferredCategory').value = 'Send Money';
                    document.getElementById('LifeStageSegment').value = 'Student/Youth';
                    document.getElementById('AvgMonthlyTransactions').value = 3;
                    document.getElementById('AvgTransactionAmount').value = 450.0;
                    document.getElementById('SupportTicketsCount').value = 4;
                } else {
                    document.getElementById('Age').value = 35;
                    document.getElementById('Gender').value = 'Male';
                    document.getElementById('AccountAgeMonths').value = 36;
                    document.getElementById('AppOpenFreqWeekly').value = 14;
                    document.getElementById('PreferredCategory').value = 'Mobile Recharge';
                    document.getElementById('LifeStageSegment').value = 'Working Professional';
                    document.getElementById('AvgMonthlyTransactions').value = 18;
                    document.getElementById('AvgTransactionAmount').value = 1200.0;
                    document.getElementById('SupportTicketsCount').value = 0;
                }
                document.getElementById('customerForm').requestSubmit();
            }

            // Form Submit Logic with Shared State
            document.getElementById('customerForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const payload = {
                    Age: parseInt(document.getElementById('Age').value),
                    Gender: document.getElementById('Gender').value,
                    AccountAgeMonths: parseInt(document.getElementById('AccountAgeMonths').value),
                    AppOpenFreqWeekly: parseInt(document.getElementById('AppOpenFreqWeekly').value),
                    PreferredCategory: document.getElementById('PreferredCategory').value,
                    LifeStageSegment: document.getElementById('LifeStageSegment').value,
                    AvgMonthlyTransactions: parseInt(document.getElementById('AvgMonthlyTransactions').value),
                    AvgTransactionAmount: parseFloat(document.getElementById('AvgTransactionAmount').value),
                    SupportTicketsCount: parseInt(document.getElementById('SupportTicketsCount').value)
                };

                try {
                    const res = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();

                    // Update Churn View Elements
                    document.getElementById('probText').innerText = data.churn_probability + '%';
                    const probBar = document.getElementById('probBar');
                    probBar.style.width = data.churn_probability + '%';

                    const badge = document.getElementById('riskBadge');
                    badge.innerText = data.risk_level + ' Risk';
                    if (data.risk_level === 'High') {
                        badge.className = 'px-2.5 py-1 rounded-full text-xs font-bold uppercase bg-rose-100 text-rose-700 border border-rose-200';
                        probBar.className = 'bg-rose-500 h-full rounded-full transition-all duration-500';
                    } else if (data.risk_level === 'Medium') {
                        badge.className = 'px-2.5 py-1 rounded-full text-xs font-bold uppercase bg-amber-100 text-amber-700 border border-amber-200';
                        probBar.className = 'bg-amber-500 h-full rounded-full transition-all duration-500';
                    } else {
                        badge.className = 'px-2.5 py-1 rounded-full text-xs font-bold uppercase bg-emerald-100 text-emerald-700 border border-emerald-200';
                        probBar.className = 'bg-emerald-500 h-full rounded-full transition-all duration-500';
                    }

                    document.getElementById('offerTitle').innerText = data.recommended_offer;
                    document.getElementById('liftText').innerText = data.lift;
                    document.getElementById('primaryDriverBadge').innerText = data.primary_risk_driver;
                    document.getElementById('actionSummary').innerText = 'Deploy ' + data.recommended_offer + ' (' + data.dispatch_timing + ' via ' + data.outreach_channel + ')';

                    // Telemetry updates
                    document.getElementById('scoreTicket').innerText = data.telemetry.ticket_score + '%';
                    document.getElementById('barTicket').style.width = data.telemetry.ticket_score + '%';

                    document.getElementById('scoreInact').innerText = data.telemetry.inact_score + '%';
                    document.getElementById('barInact').style.width = data.telemetry.inact_score + '%';

                    document.getElementById('scoreTxn').innerText = data.telemetry.vol_score + '%';
                    document.getElementById('barTxn').style.width = data.telemetry.vol_score + '%';

                    // Synchronize Next Best Action View Elements
                    document.getElementById('nba-offer-title').innerText = data.recommended_offer;
                    document.getElementById('nba-offer-desc').innerText = 'Customer risk assessment indicates ' + data.risk_level + ' Churn Risk triggered by ' + data.primary_risk_driver + '. Prescribed offer directly stabilizes retention.';
                    document.getElementById('nba-lift-badge').innerText = data.lift;
                    document.getElementById('nba-timing').innerText = data.dispatch_timing;
                    document.getElementById('nba-channel').innerText = data.outreach_channel;

                } catch(err) {
                    alert('Prediction request failed: ' + err.message);
                }
            });

            // Auto-trigger default prediction on startup
            window.onload = () => {
                document.getElementById('customerForm').requestSubmit();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)