import os
import json
import logging
import requests

logger = logging.getLogger(__name__)

SEVERITY_EMOJI = {
    "critical": "🔴",
    "warning": "🟡",
    "info": "🔵",
    "success": "✅",
}

def send_slack_alert(message: str, severity: str = "warning", title: str = None):
    """
    Send a formatted Slack alert using Block Kit.
    Falls back to logging if SLACK_WEBHOOK_URL is not set.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning(f"[SLACK-STUB] {message}")
        return

    emoji = SEVERITY_EMOJI.get(severity, "🟡")
    alert_title = title or f"{emoji} MLOps Alert"

    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": alert_title,
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{severity.upper()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\nFIRING"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{message}"
                }
            },
            {
                "type": "divider"
            }
        ],
        "attachments": [
            {
                "color": "#FF0000" if severity == "critical" else "#FFA500" if severity == "warning" else "#36a64f",
                "fallback": message
            }
        ]
    }

    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code != 200:
            logger.error(f"[SLACK] Failed to send alert: {response.status_code} {response.text}")
        else:
            logger.info(f"[SLACK] Alert sent: {message[:60]}...")
    except Exception as e:
        logger.error(f"[SLACK] Exception sending alert: {e}")
