"""
Meeting Prep Agent — Zalopay BD/AM
Input: ten_merchant + segment
Output: tin tuc merchant, talking points, cau hoi goi y
"""

import argparse
import asyncio
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

DEFAULT_BASE_URL = "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1"
DEFAULT_MODEL = "minimax/minimax-m2.5"

SEGMENTS = {
    "general":             "Tổng quan",
    "ecommerce":           "E-commerce & Marketplace",
    "retail_fnb":          "Chuỗi bán lẻ & F&B",
    "super_app_fintech":   "Super App & Fintech",
    "bank_finance":        "Ngân hàng & Tổ chức tài chính",
    "intl_merchant":       "Merchant quốc tế",
    "telecom_utility":     "Viễn thông & Tiện ích",
    "travel_hospitality":  "Du lịch & Lữ hành",
    "healthcare_edu":      "Y tế & Giáo dục",
    "government_public":   "Chính phủ & Dịch vụ công",
}

# Search keywords per segment — narrows industry context in web search
SEGMENT_SEARCH_FOCUS = {
    "general":            "fintech thanh toán Việt Nam",
    "ecommerce":          "e-commerce marketplace thương mại điện tử checkout",
    "retail_fnb":         "bán lẻ chuỗi F&B retail POS loyalty",
    "super_app_fintech":  "fintech ví điện tử super app payment",
    "bank_finance":       "ngân hàng tài chính BNPL embedded finance",
    "intl_merchant":      "merchant quốc tế cross-border local payment",
    "telecom_utility":    "viễn thông tiện ích bill payment định kỳ",
    "travel_hospitality": "du lịch lữ hành OTA hospitality booking",
    "healthcare_edu":     "y tế giáo dục bệnh viện trường học thanh toán",
    "government_public":  "dịch vụ công chính phủ eGov thuế hành chính",
}

SYSTEM_PROMPT = """Bạn là trợ lý BD/AM chuyên nghiệp của Zalopay, hỗ trợ chuẩn bị tài liệu cho buổi gặp merchant.
Nhiệm vụ: Dựa trên thông tin tìm kiếm được, tạo briefing ngắn gọn, thực tế và có thể dùng ngay.
Viết bằng tiếng Việt, súc tích, dùng bullet points.
Không bịa đặt thông tin. Nếu không tìm thấy tin tức cụ thể về merchant, nói rõ và tập trung vào ngành.
Không đề cập đến Zalopay trong nội dung phân tích — chỉ tập trung vào merchant và thị trường."""

def build_client():
    from openai import OpenAI
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        sys.exit("Loi: chua thiet lap LLM_API_KEY.")
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def web_search(query: str, max_results: int = 4) -> str:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "Khong tim thay ket qua."
        lines = []
        for r in results:
            title = r.get("title", "")
            url = r.get("href", "")
            content = r.get("body", "")
            lines.append(f"- {title} ({url})\n  {content}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Loi web search: {exc}"


def build_prompt(merchant: str, segment_key: str, search_results: dict) -> str:
    segment_label = SEGMENTS.get(segment_key, "Tổng quan")
    today = datetime.now().strftime("%d/%m/%Y")

    news_merchant = search_results.get("news_merchant", "")
    news_industry = search_results.get("news_industry", "")
    news_market = search_results.get("news_market", "")

    return f"""Ngày: {today}
Merchant cần gặp: {merchant}
Segment: {segment_label}

=== TIN TỨC VỀ MERCHANT ===
{news_merchant}

=== TIN TỨC NGÀNH / CATEGORY ===
{news_industry}

=== XU HƯỚNG THỊ TRƯỜNG ===
{news_market}

---
Hãy tạo briefing chuẩn bị cuộc gặp gồm 3 phần:

## 1. Tổng quan về {merchant}
- Tình hình hiện tại (từ tin tức tìm được)
- Điểm đáng chú ý gần đây
- Nếu không có tin tức cụ thể, ghi rõ "Chưa tìm thấy tin tức gần đây" và mô tả ngành chung

## 2. Talking Points (3–5 điểm)
- Các điểm có thể mở đầu cuộc trò chuyện
- Liên kết xu hướng thị trường với nhu cầu của merchant thuộc segment {segment_label}
- Cụ thể, thực tế, có thể dùng ngay

## 3. Câu hỏi gợi ý để hỏi đối tác (4–6 câu)
- Câu hỏi khám phá nhu cầu thực tế
- Câu hỏi về pain points hiện tại
- Câu hỏi về kế hoạch phát triển
- Phù hợp với segment {segment_label}"""


def run_meeting_prep(client, model: str, merchant: str, segment: str = "general") -> str:
    print(f"[INFO] Chuan bi meeting prep cho: {merchant} | segment: {segment}")

    focus = SEGMENT_SEARCH_FOCUS.get(segment, "fintech thanh toán")
    queries = {
        "news_merchant": f"{merchant} tin tức mới nhất 2026",
        "news_industry": f"{focus} Vietnam xu hướng 2026",
        "news_market":   f"{merchant} {focus} thị trường Việt Nam 2026",
    }

    search_results = {}
    for key, query in queries.items():
        print(f"[INFO] Searching: {query}")
        search_results[key] = web_search(query, max_results=4)

    user_prompt = build_prompt(merchant, segment, search_results)

    print("[INFO] Dang tong hop voi LLM...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=4000,
    )
    return response.choices[0].message.content or ""


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meeting Prep Agent | Zalopay BD</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background:#F2F4F8;color:#1A1A2E;min-height:100vh;line-height:1.5}
.topnav{background:#fff;padding:0 28px;height:54px;display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:100;border-bottom:1px solid #E3E8F0;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.nav-logo{display:flex;align-items:center;text-decoration:none}
.nav-logo img{height:28px;width:auto;display:block}
.nav-sep{width:1px;height:20px;background:rgba(0,0,0,.15);margin:0 4px}
.nav-prod{font-size:12px;font-weight:500;color:#9CA3AF;letter-spacing:.2px}
.hero{background:linear-gradient(135deg,#071524 0%,#0D1B2A 45%,#081E10 100%);color:#fff;padding:48px 20px 52px;text-align:center;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 60% 80% at 20% 50%,rgba(0,207,106,.07) 0%,transparent 70%),radial-gradient(ellipse 50% 70% at 80% 30%,rgba(0,195,201,.06) 0%,transparent 70%);pointer-events:none}
.hero-badge{position:relative;display:inline-flex;align-items:center;gap:7px;background:rgba(0,207,106,.1);border:1px solid rgba(0,207,106,.28);border-radius:99px;font-size:11px;font-weight:600;letter-spacing:.9px;text-transform:uppercase;padding:4px 14px;margin-bottom:18px;color:#00CF6A}
.badge-dot{width:6px;height:6px;border-radius:50%;background:#00CF6A;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
.hero h1{position:relative;font-size:28px;font-weight:800;margin-bottom:10px;letter-spacing:-.5px;line-height:1.2}
.hero-sub{position:relative;font-size:14px;opacity:.65;max-width:520px;margin:0 auto 24px;line-height:1.7}
.hero-chips{position:relative;display:flex;flex-wrap:wrap;gap:7px;justify-content:center;max-width:600px;margin:0 auto}
.topic-chip{background:rgba(0,207,106,.14);border:1px solid rgba(0,207,106,.45);border-radius:8px;padding:5px 12px;font-size:12px;font-weight:500;color:#00CF6A;cursor:default;user-select:none}
.container{max-width:960px;margin:0 auto;padding:24px 16px 60px}
.layout{display:grid;grid-template-columns:1fr 280px;gap:16px;align-items:start}
@media(max-width:720px){.layout{grid-template-columns:1fr}}
.card{background:#fff;border-radius:14px;padding:22px;border:1px solid #E3E8F0;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.sec-label{font-size:10px;font-weight:700;color:#00A855;text-transform:uppercase;letter-spacing:.9px;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.sec-label::before{content:'';display:block;width:3px;height:12px;background:linear-gradient(180deg,#00CF6A,#00C3C9);border-radius:2px}
.field{margin-bottom:14px}
.field:last-child{margin-bottom:0}
.field label{display:block;font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px}
.field-note{font-size:11px;color:#9CA3AF;margin-top:5px}
.seg-hint{font-size:11px;color:#374151;margin-top:6px;padding:6px 10px;background:#F0FFF8;border-radius:6px;border-left:2px solid #00CF6A;line-height:1.6;min-height:20px}
select,input[type=text]{width:100%;padding:9px 12px;border:1.5px solid #E3E8F0;border-radius:8px;font-size:14px;color:#1A1A2E;background:#fff;transition:border-color .15s,box-shadow .15s}
select{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='%236B7280' d='M6 8L0 0h12z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;cursor:pointer;padding-right:32px}
select:focus,input[type=text]:focus{outline:none;border-color:#00CF6A;box-shadow:0 0 0 3px rgba(0,207,106,.1)}
.sidebar-card{background:#fff;border-radius:14px;padding:22px;border:1px solid #E3E8F0;box-shadow:0 1px 3px rgba(0,0,0,.05);position:sticky;top:68px}
.run-btn{display:block;width:100%;background:linear-gradient(135deg,#00CF6A 0%,#00C3C9 100%);color:#071524;border:none;padding:13px;border-radius:10px;font-size:15px;font-weight:700;cursor:pointer;transition:all .2s;letter-spacing:.2px}
.run-btn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 4px 14px rgba(0,207,106,.3)}
.run-btn:active:not(:disabled){transform:translateY(0)}
.run-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.progress{display:none;margin-top:14px}
.pbar-track{height:3px;background:#E3E8F0;border-radius:2px;overflow:hidden}
.pbar-fill{height:100%;width:40%;background:linear-gradient(90deg,#00CF6A,#00C3C9);border-radius:2px;animation:bar 1.6s ease-in-out infinite}
@keyframes bar{0%{transform:translateX(-150%)}100%{transform:translateX(350%)}}
.pstatus{font-size:12px;color:#6B7280;margin-top:9px;display:flex;align-items:center;gap:7px}
.pdot{width:6px;height:6px;border-radius:50%;background:#00CF6A;animation:blink 1.4s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.err{display:none;margin-top:12px;background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;padding:10px 13px;color:#B91C1C;font-size:12px}
.meta-list{margin-top:16px;padding-top:14px;border-top:1px solid #F2F4F8;display:flex;flex-direction:column;gap:7px}
.meta-item{display:flex;align-items:center;gap:9px;font-size:12px;color:#6B7280}
.meta-dot{width:7px;height:7px;border-radius:50%;background:#D1D5DB;flex-shrink:0;transition:background .2s}
.meta-dot.on{background:#00CF6A}
.result-card{display:none}
.result-top{display:flex;justify-content:space-between;align-items:flex-start;padding-bottom:13px;border-bottom:1px solid #F2F4F8;margin-bottom:18px}
.result-top h2{font-size:16px;font-weight:700;color:#1A1A2E}
.result-meta{font-size:12px;color:#9CA3AF;margin-top:3px}
.dl-btn{flex-shrink:0;background:#F2F4F8;color:#374151;border:1.5px solid #E3E8F0;padding:7px 14px;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s;margin-left:12px}
.dl-btn:hover{background:#F0FFF8;border-color:#00CF6A;color:#00875A}
#report{font-size:14px;line-height:1.8;color:#1F2937}
#report h1{font-size:20px;font-weight:800;margin:0 0 14px;color:#1A1A2E}
#report h2{font-size:15px;font-weight:700;margin:24px 0 9px;color:#00875A;border-bottom:1px solid #E8F8F0;padding-bottom:6px}
#report h3{font-size:13px;font-weight:700;margin:16px 0 7px;color:#374151}
#report ul{padding-left:18px}
#report li{margin-bottom:9px}
#report p{margin-bottom:9px}
#report a{color:#00875A;text-decoration:none;border-bottom:1px dotted #00CF6A}
#report a:hover{border-bottom-style:solid}
#report strong{font-weight:700;color:#1A1A2E}
#report blockquote{border-left:3px solid #00CF6A;margin:12px 0;padding:9px 14px;background:#F0FFF8;border-radius:0 8px 8px 0;color:#374151}
#report code{background:#F2F4F8;padding:2px 6px;border-radius:4px;font-size:12px;font-family:monospace}
</style>
</head>
<body>
<nav class="topnav">
  <a href="/" class="nav-logo"><img src="data:image/png;base64,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" alt="Zalopay"></a>
  <div class="nav-sep"></div>
  <span class="nav-prod">Meeting Prep Agent</span>
</nav>
<div class="hero">
  <div class="hero-badge"><div class="badge-dot"></div>BD / AM Assistant</div>
  <h1>Meeting Prep Agent</h1>
  <p class="hero-sub">Nhập tên merchant — agent tự tìm kiếm tin tức, phân tích bối cảnh và tạo briefing chuẩn bị cuộc gặp.</p>
  <div class="hero-chips">
    <span class="topic-chip">📰 Tin tức merchant</span>
    <span class="topic-chip">💬 Talking points</span>
    <span class="topic-chip">❓ Câu hỏi gợi ý</span>
  </div>
</div>
<div class="container">
  <div class="layout">
    <div>
      <div class="card">
        <div class="sec-label">Thông tin cuộc gặp</div>
        <div class="field">
          <label>🏢 Tên merchant / đối tác</label>
          <input type="text" id="merchant" placeholder="VD: Shopee, Grab, MoMo, VinCommerce..." autocomplete="off">
        </div>
        <div class="field">
          <label>🗂 Segment BD</label>
          <select id="segment" onchange="updateSegDesc()">
            <option value="general">Tổng quan</option>
            <optgroup label="Nền tảng số">
              <option value="ecommerce">E-commerce &amp; Marketplace</option>
              <option value="super_app_fintech">Super App &amp; Fintech</option>
            </optgroup>
            <optgroup label="Thương mại &amp; Dịch vụ">
              <option value="retail_fnb">Chuỗi bán lẻ &amp; F&amp;B</option>
              <option value="travel_hospitality">Du lịch &amp; Lữ hành</option>
              <option value="telecom_utility">Viễn thông &amp; Tiện ích</option>
            </optgroup>
            <optgroup label="Tài chính">
              <option value="bank_finance">Ngân hàng &amp; Tổ chức tài chính</option>
              <option value="intl_merchant">Merchant quốc tế</option>
            </optgroup>
            <optgroup label="Khác">
              <option value="healthcare_edu">Y tế &amp; Giáo dục</option>
              <option value="government_public">Chính phủ &amp; Dịch vụ công</option>
            </optgroup>
          </select>
          <div id="segDesc" class="seg-hint">Tổng hợp chung — không giới hạn theo ngành cụ thể</div>
        </div>
      </div>
      <div class="card result-card" id="resultCard">
        <div class="result-top">
          <div>
            <h2 id="rTitle">Briefing</h2>
            <div class="result-meta" id="rMeta"></div>
          </div>
          <button class="dl-btn" onclick="dlReport()">&#8595; Tải .md</button>
        </div>
        <div id="report"></div>
      </div>
    </div>
    <div>
      <div class="sidebar-card">
        <button class="run-btn" id="runBtn" onclick="runPrep()">⚡ Tạo briefing</button>
        <div class="progress" id="progress">
          <div class="pbar-track"><div class="pbar-fill"></div></div>
          <div class="pstatus"><div class="pdot"></div><span id="pmsg">Đang khởi tạo...</span></div>
        </div>
        <div class="err" id="errBox"></div>
        <div class="meta-list">
          <div class="meta-item"><div class="meta-dot" id="mDotMerchant"></div><span id="mMerchant">Chưa nhập merchant</span></div>
          <div class="meta-item"><div class="meta-dot on" id="mDotSeg"></div><span id="mSeg">Tổng quan</span></div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const SEG_LABEL={'general':'Tổng quan','ecommerce':'E-commerce & Marketplace','retail_fnb':'Chuỗi bán lẻ & F&B','super_app_fintech':'Super App & Fintech','bank_finance':'Ngân hàng & Tổ chức tài chính','intl_merchant':'Merchant quốc tế','telecom_utility':'Viễn thông & Tiện ích','travel_hospitality':'Du lịch & Lữ hành','healthcare_edu':'Y tế & Giáo dục','government_public':'Chính phủ & Dịch vụ công'};
const SEG_DESC={'general':'Tổng hợp chung — phù hợp khi chưa rõ loại đối tác hoặc muốn góc nhìn rộng','ecommerce':'Shopee, Lazada, Tiki, TikTok Shop — checkout conversion, phí giao dịch, cạnh tranh ví','retail_fnb':'VinMart, Circle K, Highlands, CGV — POS tại quầy, loyalty, tích điểm, off-peak activation','super_app_fintech':'MoMo, VNPay, ShopeePay, Moca — co-branding, API integration, hợp tác hệ sinh thái','bank_finance':'BIDV, VPBank, Techcombank — embedded finance, BNPL, bancassurance, co-lending','intl_merchant':'Google, Apple, Meta Ads, Agoda, Booking — cross-border, local payment acceptance, FCT','telecom_utility':'Viettel, VNPT, EVN, cấp nước/gas — bill payment định kỳ, direct billing, B2B2C','travel_hospitality':'Vietnam Airlines, Bamboo, Vinpearl, OTA — checkout đặt chỗ, BNPL travel, loyalty miles','healthcare_edu':'Bệnh viện, phòng khám, trường học, edtech — học phí/viện phí định kỳ, thanh toán không tiền mặt','government_public':'Cục Thuế, DVCQG, phạt hành chính — Đề án 06, eGov payment, quy định NHNN'};
const MSGS=['Đang tìm kiếm tin tức về merchant...','Đang phân tích bối cảnh ngành...','Đang tổng hợp xu hướng thị trường...','Đang xây dựng talking points...','Đang soạn câu hỏi gợi ý...','Hoàn thiện briefing, sắp xong...'];
let mdContent='';
const sl=ms=>new Promise(r=>setTimeout(r,ms));
function updateSegDesc(){
  const v=document.getElementById('segment').value;
  document.getElementById('segDesc').textContent=SEG_DESC[v]||'';
  document.getElementById('mSeg').textContent=SEG_LABEL[v]||v;
}
document.getElementById('merchant').addEventListener('input',function(){
  const v=this.value.trim();
  const dot=document.getElementById('mDotMerchant');
  document.getElementById('mMerchant').textContent=v||'Chưa nhập merchant';
  dot.classList.toggle('on',!!v);
});
async function runPrep(){
  const merchant=document.getElementById('merchant').value.trim();
  const segment=document.getElementById('segment').value;
  if(!merchant){document.getElementById('merchant').focus();showErr('Vui lòng nhập tên merchant.');return}
  const btn=document.getElementById('runBtn'),prog=document.getElementById('progress'),err=document.getElementById('errBox'),rc=document.getElementById('resultCard');
  btn.disabled=true;rc.style.display='none';err.style.display='none';prog.style.display='block';setMsg(0);
  try{
    const r=await fetch('/invoke',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({merchant,segment})});
    const d=await r.json();
    if(!d.job_id)throw new Error(d.error||'Không nhận được job_id');
    await poll(d.job_id,merchant,segment);
  }catch(e){showErr(e.message);prog.style.display='none';btn.disabled=false}
}
let msgIdx=0;
function setMsg(i){msgIdx=i;document.getElementById('pmsg').textContent=MSGS[i%MSGS.length]}
function showErr(m){const b=document.getElementById('errBox');b.textContent='⚠ '+m;b.style.display='block'}
async function poll(id,merchant,segment){
  const btn=document.getElementById('runBtn'),prog=document.getElementById('progress');
  let t=0;
  while(true){
    setMsg(t++);await sl(5000);
    const r=await fetch('/result/'+id);const d=await r.json();
    if(d.status==='done'){prog.style.display='none';btn.disabled=false;showRpt(d.output,merchant,segment);return}
    if(d.status==='error'){prog.style.display='none';showErr(d.output||'Lỗi xử lý');btn.disabled=false;return}
  }
}
function showRpt(content,merchant,segment){
  mdContent=content;
  document.getElementById('rTitle').textContent='Briefing: '+merchant;
  document.getElementById('rMeta').textContent=(SEG_LABEL[segment]||segment)+' · '+new Date().toLocaleDateString('vi-VN');
  document.getElementById('report').innerHTML=marked.parse(content);
  const c=document.getElementById('resultCard');c.style.display='block';
  c.scrollIntoView({behavior:'smooth',block:'start'});
}
function dlReport(){
  const merchant=document.getElementById('merchant').value.trim()||'merchant';
  const b=new Blob([mdContent],{type:'text/markdown;charset=utf-8'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='briefing_'+merchant.toLowerCase().replace(/\s+/g,'_')+'_'+new Date().toISOString().slice(0,10)+'.md';a.click();
}
updateSegDesc();
</script>
</body>
</html>"""


def serve(args):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse
        import uvicorn
    except ImportError:
        sys.exit("Loi: can fastapi va uvicorn.")

    app = FastAPI(title="Meeting Prep Agent — Zalopay BD")
    client = build_client()
    default_model = args.model
    executor = ThreadPoolExecutor(max_workers=4)
    jobs: dict = {}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/invoke")
    async def invoke(request: Request):
        payload = {}
        try:
            payload = await request.json()
        except Exception:
            pass

        merchant = (payload.get("merchant") or "").strip() if isinstance(payload, dict) else ""
        if not merchant:
            return {"error": "Thieu truong 'merchant'. Vi du: {\"merchant\": \"Shopee\", \"segment\": \"key_merchant_retail\"}"}

        segment = payload.get("segment", "general") if isinstance(payload, dict) else "general"
        if segment not in SEGMENTS:
            segment = "general"

        model = (payload.get("model") if isinstance(payload, dict) else None) or default_model

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "running", "merchant": merchant, "segment": segment, "result": None}

        loop = asyncio.get_event_loop()

        def _run():
            try:
                report = run_meeting_prep(client, model=model, merchant=merchant, segment=segment)
                jobs[job_id]["result"] = report
                jobs[job_id]["status"] = "done"
            except Exception as e:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["result"] = str(e)

        loop.run_in_executor(executor, _run)

        return {
            "job_id": job_id,
            "status": "running",
            "merchant": merchant,
            "segment": segment,
            "message": "Dang chuan bi briefing. Goi GET /result/{job_id} sau ~30 giay.",
        }

    @app.get("/result/{job_id}")
    async def get_result(job_id: str):
        if job_id not in jobs:
            return {"error": "job_id khong ton tai"}
        job = jobs[job_id]
        if job["status"] == "running":
            return {"job_id": job_id, "status": "running", "message": "Dang xu ly..."}
        return {
            "job_id": job_id,
            "status": job["status"],
            "merchant": job["merchant"],
            "segment": job["segment"],
            "output": job["result"],
        }

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    parser = argparse.ArgumentParser(description="Meeting Prep Agent — Zalopay BD/AM")
    parser.add_argument("--merchant", type=str, default="", help="Ten merchant can gap")
    parser.add_argument("--segment", type=str, default="general", choices=sorted(SEGMENTS.keys()))
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--serve", action="store_true", help="Chay web server (FastAPI)")
    args = parser.parse_args()

    if args.serve:
        serve(args)
        return

    client = build_client()
    result = run_meeting_prep(client, model=args.model, merchant=args.merchant, segment=args.segment)
    print(result)


if __name__ == "__main__":
    main()
