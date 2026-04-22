# alerts.py - Already good, no changes needed
def send_alert(risk):
    if risk == 1:
        return "🚨 HIGH RISK ALERT!"
    return "Safe"
