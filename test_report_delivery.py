from output.report_generator import generate_report
from output.notifier_email import send_email_via_smtp
from output.notifier_slack import send_slack_message
from output.notifier_sms import send_sms_via_twilio

mock_data = {
    "price": 502.10,
    "change": "+2.1%",
    "rsi": 72,
    "macd_signal": "bullish",
    "news_sentiment": "80% positive",
    "reddit_sentiment": "70% positive",
    "sec_summary": "No new risks in latest 10-Q",
    "final_signal": "HOLD"
}

if __name__ == "__main__":
    # 1. Generate report
    report = generate_report(mock_data)
    print(report)

    # 2. Send via email
    send_email_via_smtp(report)

    # 3. Send via Slack
    send_slack_message(report)

    # 4. Send via SMS
    send_sms_via_twilio(report)