import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Student Sathi", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")
BASE=Path(__file__).parent
DATA=BASE/"student_support_demo_data.csv"
df=pd.read_csv(DATA)
FEATURES=["attendance_pct","overall_score","subject_average","weakest_subject_score","failed_subjects","study_hours_week","commute_hours_day","financial_stress","work_hours_week","parent_support"]

@st.cache_resource
def train_model():
    X=df[FEATURES]; y=df["support_needed"]
    m=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(max_iter=1500,class_weight="balanced"))])
    m.fit(X,y); return m
model=train_model()

if "student_profiles" not in st.session_state: st.session_state.student_profiles=[]
if "last_result" not in st.session_state: st.session_state.last_result=None
if "counseling_requests" not in st.session_state: st.session_state.counseling_requests=[]

# ---- polished visual language ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--ink:#17243a;--muted:#68758a;--teal:#1c9a8c;--teal2:#c9f2eb;--violet:#6c63ff;--peach:#ffb36b;--rose:#ff7b87;--bg:#f6f8fb;--white:#fff;--line:#e5eaf1}
.stApp{background:radial-gradient(circle at 8% 2%,#e8fbf7 0,transparent 25%),radial-gradient(circle at 94% 6%,#eeeaff 0,transparent 23%),linear-gradient(180deg,#f8fafc,#fff 42%,#f7f9fc);font-family:'DM Sans',sans-serif;color:var(--ink)}
.block-container{max-width:1180px;padding:1.2rem 1.5rem 3rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:var(--ink)}
.navbar{display:flex;align-items:center;justify-content:space-between;background:rgba(255,255,255,.86);backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:18px;padding:.65rem 1rem;position:sticky;top:.4rem;z-index:99;box-shadow:0 10px 30px rgba(20,35,55,.06)}
.brand{font-family:'Space Grotesk';font-weight:700;font-size:1.1rem}.brand span{color:var(--teal)}
.eyebrow{letter-spacing:.12em;text-transform:uppercase;font-weight:700;font-size:.75rem;color:var(--teal)}
.hero{margin-top:1rem;border-radius:30px;padding:2.2rem 2.4rem;background:linear-gradient(135deg,#17243a 0%,#263b59 55%,#159b8e 120%);color:#fff;box-shadow:0 24px 55px rgba(23,36,58,.16);overflow:hidden;position:relative}
.hero:after{content:'';position:absolute;width:250px;height:250px;border-radius:50%;right:-70px;top:-90px;background:rgba(255,255,255,.08);box-shadow:-90px 160px 0 20px rgba(255,255,255,.05)}
.hero h1{font-size:3.5rem;color:#fff;margin:.2rem 0 .4rem;letter-spacing:-.04em}.hero p{font-size:1.12rem;color:#dce7f1;max-width:690px}
.hero-pill{display:inline-block;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);padding:.4rem .7rem;border-radius:999px;font-weight:600;font-size:.82rem}
.card{background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:22px;padding:1.25rem 1.35rem;box-shadow:0 10px 30px rgba(30,45,65,.055);height:100%}
.card:hover{box-shadow:0 15px 35px rgba(30,45,65,.08)}
.soft{background:linear-gradient(135deg,#f0fbf9,#fff);border:1px solid #d7f1eb}.purple{background:linear-gradient(135deg,#f3f1ff,#fff);border:1px solid #e5e0ff}.peach{background:linear-gradient(135deg,#fff7ed,#fff);border:1px solid #ffe7ca}
.kpi{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1rem 1.1rem}.kpi .n{font-family:'Space Grotesk';font-size:1.9rem;font-weight:700}.kpi .l{color:var(--muted);font-size:.82rem}
.step{display:flex;gap:.8rem;align-items:flex-start}.dot{min-width:34px;height:34px;border-radius:50%;display:grid;place-items:center;background:var(--teal2);color:#13796f;font-weight:700}
.callout{border-left:4px solid var(--teal);padding:.8rem 1rem;background:#f1fbf9;border-radius:0 14px 14px 0}.tiny{font-size:.82rem;color:var(--muted)}
.result{border-radius:22px;padding:1.2rem;background:linear-gradient(135deg,#eafff9,#fff)}
.footer{color:#8490a2;font-size:.78rem;text-align:center;padding-top:2rem}
.stButton>button{border-radius:13px;border:0;background:linear-gradient(135deg,#1c9a8c,#177c75);color:#fff;font-weight:700;padding:.7rem 1rem;box-shadow:0 8px 18px rgba(28,154,140,.18)}
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:.8rem}
</style>
""",unsafe_allow_html=True)

# Top nav — only three working areas
nav=st.radio("",["🏠 Home","📝 Student Check-in","👩‍🏫 Mentor Hub"],horizontal=True,label_visibility="collapsed")

# ---- HOME ----
if nav=="🏠 Home":
    st.markdown('<div class="hero"><span class="hero-pill">EARLY SUPPORT • HUMAN FIRST</span><h1>Student Sathi</h1><p>A simple bridge between a student who may be struggling and the person who can actually help — before the problem becomes bigger.</p></div>',unsafe_allow_html=True)
    st.markdown("""<div style="display:flex;justify-content:center;margin:18px 0 4px"><svg width="900" height="150" viewBox="0 0 900 150" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="g" x1="0" x2="1"><stop stop-color="#1c9a8c"/><stop offset="1" stop-color="#6c63ff"/></linearGradient></defs><path d="M80 85 C230 20 310 130 450 75 S680 25 820 75" fill="none" stroke="#dbe4ee" stroke-width="8" stroke-linecap="round"/><path d="M80 85 C230 20 310 130 450 75 S680 25 820 75" fill="none" stroke="url(#g)" stroke-width="4" stroke-dasharray="10 14" stroke-linecap="round"><animate attributeName="stroke-dashoffset" from="0" to="-48" dur="2s" repeatCount="indefinite"/></path><circle cx="80" cy="85" r="18" fill="#1c9a8c"/><circle cx="450" cy="75" r="18" fill="#6c63ff"/><circle cx="820" cy="75" r="18" fill="#ffb36b"/><text x="44" y="125" font-size="15" fill="#68758a" font-family="Arial">Student</text><text x="410" y="118" font-size="15" fill="#68758a" font-family="Arial">Mentor</text><text x="780" y="118" font-size="15" fill="#68758a" font-family="Arial">Teacher</text></svg></div>""",unsafe_allow_html=True)
    st.write("")
    a,b,c=st.columns(3)
    for col,head,body,cls in [
        (a,"Notice early","Use a few academic and practical signals to spot where support may be useful.","soft"),
        (b,"Name the subject","Students enter the subjects that actually exist in their course. Nothing is hard-coded.","purple"),
        (c,"Connect a human","A mentor can privately route the student to the relevant subject teacher and follow up.","peach")]:
        with col: st.markdown(f'<div class="card {cls}"><div class="eyebrow">{head}</div><h3>{body.split(".")[0]}</h3><p>{body[len(body.split(".")[0])+1:]}</p></div>',unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="card"><div class="eyebrow">WHY THIS IS REAL</div><h2>Support should not begin only after a student has already fallen behind.</h2><p>Education systems collect information on enrolment, examination results, finance and infrastructure, and current education guidance increasingly emphasises actionable student-support information. Student Sathi turns that idea into a small, human-centred workflow: signal → conversation → support → follow-up.</p></div>',unsafe_allow_html=True)
    st.write("")
    c1,c2=st.columns([1.25,1])
    with c1:
        st.markdown('<div class="card"><div class="eyebrow">THE 5-STAGE DESIGN STORY</div>',unsafe_allow_html=True)
        for i,(title,txt) in enumerate([("Empathize","Understand the student, mentor and teacher experience."),("Define","Frame the gap: difficulty may stay invisible until it becomes serious."),("Ideate","Combine early signals with a respectful teacher-support workflow."),("Prototype","Build a working checker and subject-wise mentor handoff."),("Test","Check clarity, usability and whether the output leads to useful action.")],1):
            st.markdown(f'<div class="step"><div class="dot">{i}</div><div><b>{title}</b><div class="tiny">{txt}</div></div></div><div style="height:.55rem"></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card purple"><div class="eyebrow">ONE IMPORTANT RULE</div><h2>It is a support signal, not a label.</h2><p>The model does not decide whether someone will drop out. It does not punish, rank or publicly label students.</p><div class="callout"><b>Human action comes next:</b><br>mentor notices → teacher checks in privately → student gets the right guidance → support is followed up.</div></div>',unsafe_allow_html=True)
    st.caption("Demo note: the bundled model data are synthetic. A real institute should validate on anonymised local data before operational use.")

# ---- STUDENT CHECK-IN ----
elif nav=="📝 Student Check-in":
    st.markdown('<div class="hero"><span class="hero-pill">YOUR SPACE • NO FIXED SUBJECT LIST</span><h1>Tell us what is going on.</h1><p>Enter the subjects you actually study and your current marks. Student Sathi will highlight where support may be useful — without judging you.</p></div>',unsafe_allow_html=True)
    st.write("")
    with st.form("student_checkin"):
        left,right=st.columns(2)
        with left:
            name=st.text_input("Your name",placeholder="e.g., Aditi")
            st.markdown("**📅 Attendance**")
            st.caption("Approximately what percentage of classes do you attend?")
            attendance=st.slider("Attendance percentage",35,100,75,format="%d%%",label_visibility="collapsed")
            st.caption(f"Selected: **{attendance}% attendance**")
            st.markdown("**📊 Overall academic score**")
            st.caption("What is your approximate overall academic performance?")
            overall=st.slider("Overall academic score",20,100,65,format="%d%%",label_visibility="collapsed")
            st.caption(f"Selected: **{overall}%**")
            st.markdown("**📚 Subjects currently not cleared**")
            st.caption("How many subjects do you currently need to clear?")
            failed=st.slider("Subjects not cleared",0,6,0,format="%d subject(s)",label_visibility="collapsed")
            st.caption(f"Selected: **{failed} subject(s)**")
        with right:
            st.markdown("**📖 Study time outside class**")
            st.caption("Approximately how many hours do you study outside college each week?")
            study=st.slider("Study hours per week",0,35,8,format="%d hr/week",label_visibility="collapsed")
            st.caption(f"Selected: **{study} hours/week**")
            st.markdown("**🚌 Daily commute time**")
            st.caption("How many hours do you spend travelling to and from college each day?")
            commute=st.slider("Commute hours per day",0.0,5.0,1.5,0.1,format="%.1f hr/day",label_visibility="collapsed")
            st.caption(f"Selected: **{commute:.1f} hours/day**")
            st.markdown("**💰 Financial pressure**")
            st.caption("How would you describe the financial pressure you are currently experiencing?")
            financial=st.selectbox("Financial pressure level",["Low","Medium","High","Very high"],label_visibility="collapsed")
            st.caption(f"Selected: **{financial}**")
            st.markdown("**💼 Work / responsibility hours**")
            st.caption("How many hours each week go toward work or other major responsibilities?")
            work=st.slider("Work responsibility hours",0,35,5,format="%d hr/week",label_visibility="collapsed")
            st.caption(f"Selected: **{work} hours/week**")
            st.markdown("**🤝 Study support at home**")
            st.caption("Do you have someone you can regularly ask for study support?")
            family=st.selectbox("Study support",["Yes","Not regularly"],label_visibility="collapsed")
            st.caption(f"Selected: **{family}**")
        st.markdown('<div class="card soft"><div class="eyebrow">YOUR ACTUAL SUBJECTS</div><h3>Add only what your course uses</h3><p class="tiny">Type the subject name yourself. You can add as many rows as you need. This works across different degrees, institutes and semesters.</p></div>',unsafe_allow_html=True)
        subjects=st.data_editor(pd.DataFrame({"Subject name":["","",""],"Marks (%)":[None,None,None]}),num_rows="dynamic",use_container_width=True,hide_index=True,column_config={"Subject name":st.column_config.TextColumn("Subject name"),"Marks (%)":st.column_config.NumberColumn("Marks (%)",min_value=0,max_value=100,step=1)},key="dynamic_subjects")
        st.markdown('<div class="card purple"><div class="eyebrow">PRIVATE WELLBEING SUPPORT</div><h3>💙 Need someone to talk to?</h3><p>If academic pressure, anxiety, low mood, loneliness or another personal difficulty is making college feel hard, you can request a private conversation with a counsellor. You do not need to share a diagnosis or explain everything here.</p><p class="tiny"><b>Designed to reduce awkwardness:</b> the request is kept separate from the academic subject-support workflow. In a production version, it should be routed only to an authorised counsellor through secure, access-controlled systems.</p><p class="tiny"><b>Need urgent help in India?</b> Tele-MANAS provides 24x7 tele-mental health support at <b>14416</b> or <b>1800-89-14416</b>.</p></div>',unsafe_allow_html=True)
        wellbeing=st.selectbox("Private wellbeing support",["No, not right now","Yes, I would like a private counselling conversation"],index=0)
        wellbeing_mode=st.selectbox("Preferred conversation format",["Online / video","In-person at college","Either is fine"])
        wellbeing_note=st.text_area("Optional note for the counsellor (do not include passwords or highly sensitive details)",placeholder="You can simply write: “I would like someone to talk to.”",height=80)
        private=st.checkbox("I would prefer a private conversation with a mentor/teacher if academic support is needed.",value=True)
        submitted=st.form_submit_button("✨ Check where I may need support",use_container_width=True)
    if submitted:
        clean=[]
        for _,r in subjects.iterrows():
            s=str(r.get("Subject name","")).strip(); m=r.get("Marks (%)")
            if s and pd.notna(m):
                try:
                    m=float(m)
                    if 0<=m<=100: clean.append({"subject":s,"score":m})
                except: pass
        if not clean:
            st.error("Please add at least one subject name and marks.")
        elif len({x["subject"].casefold() for x in clean}) != len(clean):
            st.error("Please enter each subject only once so the mentor view stays accurate.")
        else:
            fm={"Low":0,"Medium":1,"High":2,"Very high":3}
            avg=float(np.mean([x["score"] for x in clean])); weak=min(clean,key=lambda x:x["score"])
            row=pd.DataFrame([{"attendance_pct":attendance,"overall_score":overall,"subject_average":avg,"weakest_subject_score":weak["score"],"failed_subjects":failed,"study_hours_week":study,"commute_hours_day":commute,"financial_stress":fm[financial],"work_hours_week":work,"parent_support":1 if family=="Yes" else 0}])[FEATURES]
            risk=float(model.predict_proba(row)[0,1]); score=int(round(risk*100))
            if score>=70: level="Higher support signal"; tone="It may be worth having a supportive conversation soon."
            elif score>=45: level="Some support may help"; tone="A small intervention now could prevent the issue from growing."
            else: level="No strong support signal right now"; tone="Keep checking in — a single score never tells the whole story."
            reasons=[]
            if attendance<65: reasons.append("attendance")
            if overall<50 or weak["score"]<45: reasons.append("academic performance")
            if failed>=2: reasons.append("uncleared subjects")
            if fm[financial]>=2: reasons.append("financial pressure")
            if commute>=2.5: reasons.append("travel/time pressure")
            if work>=15: reasons.append("work/responsibility load")
            if not reasons: reasons=["no single strong warning sign in this demo input"]
            profile={"student_name":name.strip() or "Student","support_signal":score,"attendance":attendance,"overall":overall,"failed":failed,"weakest_subject":weak["subject"],"weakest_score":weak["score"],"subjects":clean,"private":private,"reasons":reasons}
            if wellbeing == "Yes, I would like a private counselling conversation":
                st.session_state.counseling_requests.append({"student_name":name.strip() or "Student","mode":wellbeing_mode})
            st.session_state.student_profiles.append(profile); st.session_state.last_result=profile
            st.markdown(f'<div class="result"><div class="eyebrow">YOUR RESULT</div><h2>{level}</h2><p>{tone}</p><div class="callout"><b>Your lowest entered subject:</b> {weak["subject"]} — {weak["score"]:.0f}%<br><span class="tiny">A mentor can use this to route you to the teacher responsible for that subject.</span></div></div>',unsafe_allow_html=True)
            c1,c2,c3=st.columns(3); c1.metric("Support signal",f"{score}%"); c2.metric("Subjects entered",len(clean)); c3.metric("Lowest mark",f"{weak['score']:.0f}%")
            st.progress(risk)
            st.subheader("What may help")
            for r in reasons: st.write("•",r.capitalize())
            st.info("Your subject name is carried forward exactly as you entered it. Student Sathi does not assume a fixed subject list.")
            st.success("Next step: a class mentor can privately connect you with the relevant teacher for guidance, doubt-solving or guided practice.")
            if wellbeing == "Yes, I would like a private counselling conversation":
                st.markdown('<div class="callout"><b>💙 Private wellbeing request received.</b><br>Your request is separate from the subject-support pathway. In a production deployment, only an authorised counsellor should receive the request and arrange the conversation. You do not need to disclose personal details to your teacher or parents through this form.</div>',unsafe_allow_html=True)
            st.caption("This is a demonstration prototype using synthetic training data. The support signal is not a mental-health diagnosis. Counselling requests require a real authorised counselling service and secure access controls in a production system.")

# ---- MENTOR HUB ----
else:
    st.markdown('<div class="hero"><span class="hero-pill">INSTITUTE VIEW • HUMAN FOLLOW-UP</span><h1>Mentor Hub</h1><p>A calm, subject-wise view of who may need attention — so the right teacher can be involved without publicly labelling anyone.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>🔒 Privacy boundary:</b> wellbeing request details are not shown in this mentor view. In a real deployment, they should be routed to an authorised counsellor using authenticated, encrypted and role-based access — not shared with teachers or classmates.</div>',unsafe_allow_html=True)
    profiles=st.session_state.student_profiles
    if not profiles:
        st.write(""); st.markdown('<div class="card soft"><h2>No students in this demo session yet.</h2><p>Use <b>Student Check-in</b> to add a few student profiles. Then return here to see the subject-wise support map.</p></div>',unsafe_allow_html=True)
    else:
        total=len(profiles); higher=sum(p["support_signal"]>=70 for p in profiles); subjects={}
        for p in profiles:
            for item in p["subjects"]:
                subjects.setdefault(item["subject"].strip(),[]).append((p["student_name"],item["score"]))
        k1,k2,k3,k4=st.columns(4); k1.metric("Students reviewed",total); k2.metric("Higher support signals",higher); k3.metric("Subjects represented",len(subjects)); k4.metric("Private wellbeing requests",len(st.session_state.counseling_requests))
        st.write("")
        st.markdown('<div class="card"><div class="eyebrow">STUDENT OVERVIEW</div><p class="tiny">Use this as a private working list, not a public ranking.</p>',unsafe_allow_html=True)
        rows=[]
        for p in profiles: rows.append({"Student":p["student_name"],"Support signal":f"{p['support_signal']}%","Lowest entered subject":p["weakest_subject"],"Mark":f"{p['weakest_score']:.0f}%","Private follow-up":"Yes" if p["private"] else "Not requested"})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="card purple"><div class="eyebrow">SUBJECT-WISE HANDOFF</div><h2>Who needs the attention of which subject teacher?</h2><p class="tiny">The subject names below come from the students. A mentor can gather the names and privately contact the teacher responsible for each subject.</p>',unsafe_allow_html=True)
        for subject,students in sorted(subjects.items(),key=lambda x:x[0].lower()):
            attention=sorted([(n,s) for n,s in students if s<65],key=lambda x:x[1])
            if attention:
                st.markdown(f"### {subject}")
                for name,score in attention:
                    label="Guided support / private check-in" if score<50 else "Doubt-solving / follow-up"
                    st.write(f"• **{name}** — {score:.0f}%  →  {label}")
                st.caption(f"Mentor action: privately inform the teacher responsible for {subject} and suggest a short check-in.")
        st.markdown('</div>',unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="card soft"><div class="eyebrow">A SIMPLE CONVERSATION</div><h3>Keep it human.</h3><p>“I noticed you may be finding this subject a little difficult. Is there a specific concept, practice problem or time issue making it harder? We can work out a support plan together.”</p><p class="tiny">No public labels • no punishment • no automatic academic decision</p></div>',unsafe_allow_html=True)
        if st.button("Clear this demo session"): st.session_state.student_profiles=[]; st.session_state.last_result=None; st.rerun()

st.markdown('<div class="footer">Student Sathi • Design Thinking prototype • Synthetic demonstration data • Real deployment requires local validation, privacy controls and human review.</div>',unsafe_allow_html=True)
