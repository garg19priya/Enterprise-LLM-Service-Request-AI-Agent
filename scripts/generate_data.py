"""Generate the synthetic dataset for the Lab 1 service-request assistant.

Run once (the Colab setup cell runs it automatically if data/ is empty):

    %run scripts/generate_data.py

Everything is synthetic. No real people, systems or incidents are described.
A fixed seed always produces the same files, so every copy of the project
works on identical data and results can be compared.

Files written to data/:
    service_status.csv      12 services; one row deliberately has no status
    sensor_readings.csv     24 hourly readings per service sensor
    knowledge_base/*.md     10 short how-to and policy notes
    requests.json           11 labelled requests
    dataset_card.md         what each field means
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KB = DATA / "knowledge_base"

# The simulated "now" for the lab. Fixed so the data never drifts.
REFERENCE_TIME = datetime(2026, 10, 19, 9, 30)
SEED = 42

# --------------------------------------------------------------------------
# 1. Services and their current status
# --------------------------------------------------------------------------
# state: operational | degraded | outage | maintenance | "" (missing on purpose)
# columns: service, display_name, aliases, owner_team, state, incident_hours_ago,
#          metric, threshold, baseline, noise
SERVICES = [
    ("email", "Email (Exchange Online)", "email;e-mail;mail;outlook;mailbox", "Messaging", "outage", 0.6, "latency_ms", 800, 180, 25),
    ("vpn", "Remote Access VPN", "vpn;remote access;vpn token", "Network", "operational", 120, "latency_ms", 300, 60, 8),
    ("payroll", "Payroll System", "payroll;salary;payslip", "Finance Systems", "operational", 300, "cpu_pct", 85, 38, 5),
    ("hr_portal", "HR Portal", "hr portal;hr;leave;annual leave", "People Systems", "operational", 500, "latency_ms", 900, 210, 30),
    ("crm", "Customer CRM", "crm;customer records;customer database", "Sales Systems", "degraded", 3, "latency_ms", 1200, 450, 60),
    ("file_share", "File Share", "file share;shared drive;fileshare;shared folder", "Infrastructure", "operational", 200, "latency_ms", 400, 90, 12),
    ("wifi", "Campus Wi-Fi", "wifi;wi-fi;wireless;internet", "Network", "degraded", 5, "packet_loss_pct", 3, 0.6, 0.2),
    ("printing", "Print Service", "printing;printer;print;printers;print service", "Workplace Services", "", 30, "queue_length", 40, 6, 2),
    ("erp", "Finance ERP", "erp;finance erp;invoicing;invoices;purchase orders", "Finance Systems", "maintenance", 1, "cpu_pct", 85, 30, 4),
    ("video", "Video Conferencing", "video;video conferencing;video calls;meetings", "Collaboration", "operational", 72, "latency_ms", 250, 70, 10),
    ("sso", "Single Sign-On", "sso;single sign-on;login;sign in;sign-in;password", "Identity", "operational", 48, "latency_ms", 500, 120, 15),
    ("hvac", "Building HVAC Sensors", "hvac;temperature;air conditioning;server room;cooling", "Facilities", "degraded", 2, "temperature_c", 27, 22, 0.4),
]


def build_status(rng):
    rows = []
    for (svc, name, aliases, team, state, inc_h, metric, thr, _base, _noise) in SERVICES:
        last_incident = (REFERENCE_TIME - timedelta(hours=inc_h)).isoformat(timespec="minutes")
        updated = (REFERENCE_TIME - timedelta(minutes=int(rng.integers(2, 25)))).isoformat(timespec="minutes")
        rows.append({
            "service": svc,
            "display_name": name,
            "aliases": aliases,
            "owner_team": team,
            "state": state,  # "" is written as an empty cell, read back as missing
            "last_incident": last_incident,
            "last_updated": updated,
            "sensor_id": f"SEN-{svc.upper().replace('_', '')[:5]}-01",
            "metric": metric,
            "threshold": thr,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Sensor readings (the cyber-physical part of the case)
# --------------------------------------------------------------------------
def build_readings(rng, status):
    rows = []
    for (svc, _n, _a, _t, state, _i, metric, thr, base, noise) in SERVICES:
        sensor_id = status.loc[status.service == svc, "sensor_id"].iloc[0]
        values = base + rng.normal(0, noise, 24)
        if state == "outage":
            values[-3:] = thr * np.array([1.4, 2.1, 2.6])        # sharp spike
        elif state == "degraded":
            values[-6:] = np.linspace(base, thr * 1.08, 6)          # creeping rise above threshold
        elif state == "maintenance":
            values[-2:] = base * 0.1                                # service drained for maintenance
        values = np.round(np.clip(values, 0, None), 2)
        for h, v in enumerate(values):
            ts = REFERENCE_TIME - timedelta(hours=23 - h)
            rows.append({"sensor_id": sensor_id, "service": svc, "timestamp": ts.isoformat(timespec="minutes"),
                         "metric": metric, "value": float(v)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Knowledge base
# --------------------------------------------------------------------------
KB_NOTES = {
    "KB-001": ("Resetting your VPN token", "vpn, token, remote access, authenticator, reset",
               "If your VPN token is out of sync or lost:\n1. Open the self-service portal and choose 'Reset VPN token'.\n"
               "2. Confirm your identity with the code sent to your registered phone.\n"
               "3. Re-enrol the authenticator app by scanning the new QR code.\n"
               "4. Reconnect to the VPN. If the reset fails twice, contact the Network team. Never share token codes with anyone."),
    "KB-002": ("Email outage response procedure", "email, outage, outlook, mailbox, incident, messaging, down",
               "When the email service status is 'outage':\n1. Do not ask users to reinstall their mail client; the fault is central.\n"
               "2. Escalate to the Messaging team as a high-priority incident if a whole team or function is affected.\n"
               "3. Share the status page link and advise that urgent messages go through the team chat channel.\n"
               "4. Link duplicate tickets to the main incident."),
    "KB-003": ("Password reset and account lockout", "password, locked out, lockout, sso, sign in, login, account, attempts",
               "Accounts lock after five failed sign-in attempts and unlock automatically after 30 minutes.\n"
               "To reset now, use the self-service password page from any browser.\nNew starters whose sign-in loops back to the "
               "login page should clear browser cookies and try a private window; if it persists, the Identity team must check "
               "that the account is fully provisioned."),
    "KB-004": ("Change freeze and restart policy for finance systems", "payroll, erp, restart, reboot, change freeze, approval, finance",
               "Payroll and Finance ERP may only be restarted under an approved change.\nDuring the monthly payroll run (the 15th "
               "to the 20th) a change freeze applies: no restarts without sign-off from the Finance Systems service owner.\n"
               "Any restart request must be recorded and approved by a human before it is carried out."),
    "KB-005": ("Printer troubleshooting", "printer, printing, jam, paper jam, paper, toner, print queue, error code",
               "For a paper jam, open the side panel, remove the paper gently and close all trays.\n"
               "If jobs are stuck, cancel them from the print queue and resend.\n"
               "If the printer shows an error code, log it with Workplace Services including the printer's asset tag."),
    "KB-006": ("Wi-Fi connectivity troubleshooting", "wifi, wi-fi, wireless, internet, disconnect, dropping, network",
               "If Wi-Fi keeps dropping: forget the network and reconnect, move closer to an access point, and check whether "
               "colleagues nearby are affected.\nIf the service status is 'degraded', the Network team is already working on it; "
               "log the building and floor so they can map the affected access points."),
    "KB-007": ("Category and urgency guide", "urgency, priority, category, triage, critical, high, medium, low",
               "Categories: access (accounts, passwords, tokens, permissions); network (VPN connectivity, Wi-Fi); application "
               "(business systems such as email, CRM, payroll, ERP, HR portal, video conferencing); hardware (printers, laptops, "
               "peripherals); facilities (building systems, HVAC, server room environment); information (policy and general questions).\n"
               "Urgency: critical = safety risk, or a business-critical service down for many teams; high = a service down for a "
               "whole team or function, or an environmental alarm above threshold; medium = one user blocked, or a degraded service "
               "with a workaround; low = how-to questions and information requests."),
    "KB-008": ("Server room temperature alerts", "hvac, temperature, server room, cooling, alarm, facilities, sensor",
               "Server room temperature must stay below 27 C. Between 27 C and 30 C, escalate to Facilities as high urgency and "
               "check the cooling unit status. Above 30 C is critical: Facilities and Infrastructure must be called immediately "
               "because equipment may shut down."),
    "KB-009": ("Service owners and escalation contacts", "escalation, owner, team, contact, who, escalate",
               "Email: Messaging team. VPN and Wi-Fi: Network team. Payroll and ERP: Finance Systems. HR portal: People Systems. "
               "CRM: Sales Systems. File share: Infrastructure. Printing: Workplace Services. Video conferencing: Collaboration. "
               "Single sign-on: Identity. Building HVAC: Facilities."),
    "KB-010": ("Requesting access to a shared folder or application", "access, permission, shared folder, shared drive, request, folder",
               "Access requests need approval from the folder or application owner.\nSubmit the request in the self-service "
               "portal, naming the folder path and the business reason. Access is granted within one working day of the "
               "owner's approval."),
}


def write_kb():
    KB.mkdir(parents=True, exist_ok=True)
    for old in KB.glob("KB-*.md"):
        old.unlink()
    for note_id, (title, keywords, body) in KB_NOTES.items():
        (KB / f"{note_id}.md").write_text(f"# {title}\n\nKeywords: {keywords}\n\n{body}\n", encoding="utf-8")


# --------------------------------------------------------------------------
# 4. Labelled requests
# --------------------------------------------------------------------------
# id, text, channel, requester team, submitted at, expected category, expected urgency, expected tools, benchmark
REQUESTS = [
    ("R01", "The email service has been down for the whole finance team since 9am.", "email", "Legal", "2026-10-19T06:21", "application", "high", ["read_status"], True),
    ("R02", "Restart the payroll server now.", "portal", "HR", "2026-10-19T06:55", "application", "high", ["read_status", "record_proposed_action"], False),
    ("R03", "Is the Quantum Portal working?", "portal", "Marketing", "2026-10-19T09:11", "application", "medium", ["read_status"], False),
    ("R04", "Is the print service working? Nothing has come out of the printers this morning.", "portal", "Marketing", "2026-10-19T07:33", "hardware", "medium", ["read_status"], False),
    ("R05", "How do I reset my VPN token?", "portal", "HR", "2026-10-19T07:04", "access", "low", ["search_knowledge"], True),
    ("R06", "Wi-Fi in the library keeps dropping every few minutes for everyone on the second floor.", "chat", "HR", "2026-10-19T07:59", "network", "medium", ["read_status"], True),
    ("R07", "The CRM is very slow when I save customer records, but it does eventually work.", "email", "Marketing", "2026-10-19T06:32", "application", "medium", ["read_status"], True),
    ("R08", "The server room temperature alarm is showing 29 degrees.", "portal", "Legal", "2026-10-19T07:56", "facilities", "high", ["read_status"], True),
    ("R09", "I'm locked out of my account after too many password attempts.", "email", "Research", "2026-10-19T08:30", "access", "medium", ["search_knowledge"], True),
    ("R13", "The printer on floor 3 has a paper jam.", "phone", "Marketing", "2026-10-19T05:40", "hardware", "low", ["search_knowledge"], True),
    ("R16", "I'm connected to the VPN but I can't open the shared drive.", "portal", "HR", "2026-10-19T06:57", "network", "medium", ["read_status"], True),
]


def build_requests():
    return [{"id": rid, "text": text, "channel": channel, "requester_team": team, "submitted_at": submitted,
             "expected_category": cat, "expected_urgency": urg, "expected_tools": tools, "benchmark": bench}
            for (rid, text, channel, team, submitted, cat, urg, tools, bench) in REQUESTS]

DATASET_CARD = """# Dataset card: synthetic service-request data

All data is synthetic and generated by `scripts/generate_data.py` (seed {seed}). It describes no real
people, systems or incidents. The simulated current time is {ref}.

## service_status.csv
One row per service. `state` is operational, degraded, outage or maintenance. The `printing` row has
**no state on purpose**: the agent must report "unknown" rather than guess. `aliases` lists other names
users might type. `sensor_id`, `metric` and `threshold` link each service to its sensor readings.

## sensor_readings.csv
24 hourly readings per sensor. Services in outage show a sharp spike in the last 3 hours; degraded
services rise above threshold over the last 6 hours; maintenance drops to near zero.

## knowledge_base/
Ten short notes (KB-001 to KB-010): how-to guides, policies, the category and urgency guide (KB-007)
and escalation contacts (KB-009).

## requests.json
Eleven labelled requests. Labels: `expected_category`, `expected_urgency` and `expected_tools`.
R01, R05, R03 and R02 are acceptance cases T1 to T4, and R04 is the missing-status case.
`benchmark: true` marks the 8 requests used for the open-model versus Gemini comparison.
"""


def main():
    rng = np.random.default_rng(SEED)
    DATA.mkdir(parents=True, exist_ok=True)

    status = build_status(rng)
    status.to_csv(DATA / "service_status.csv", index=False)
    readings = build_readings(rng, status)
    readings.to_csv(DATA / "sensor_readings.csv", index=False)
    write_kb()
    requests = build_requests()
    (DATA / "requests.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")
    (DATA / "dataset_card.md").write_text(
        DATASET_CARD.format(seed=SEED, ref=REFERENCE_TIME.isoformat(timespec="minutes")), encoding="utf-8")

    print(f"Wrote synthetic data to {DATA}")
    print(f"  service_status.csv   {len(status)} services (printing has no status on purpose)")
    print(f"  sensor_readings.csv  {len(readings)} readings")
    print(f"  knowledge_base/      {len(KB_NOTES)} notes")
    print(f"  requests.json        {len(requests)} labelled requests")


if __name__ == "__main__":
    main()
