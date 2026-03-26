import os, secrets, smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, PasswordReset
from ..schemas import UserRegister, UserLogin, UserOut, Token, UserUpdate, ForgotPassword, ResetPassword
from ..auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Email helper ────────────────────────────────────────────────

def _send_email(to: str, subject: str, html: str):
    """Send email via SMTP. Configured via env vars."""
    host      = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port      = int(os.environ.get("SMTP_PORT", "587"))
    user      = os.environ.get("SMTP_USER")
    password  = os.environ.get("SMTP_PASS")
    from_name = "Stolcade Arena"

    if not user or not password:
        print(f"[email] SMTP not configured — skipping email to {to}: {subject}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"{from_name} <{user}>"
    msg["To"]      = to
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(host, port, timeout=10) as srv:
            srv.starttls()
            srv.login(user, password)
            srv.sendmail(user, to, msg.as_string())
    except Exception as e:
        print(f"[email] Failed to send to {to}: {e}")


# ── Endpoints ───────────────────────────────────────────────────

@router.post("/register", response_model=UserOut, status_code=201)
def register(body: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(400, "Username already taken")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    user = User(username=body.username, email=body.email,
                hashed_password=hash_password(body.password))
    db.add(user); db.commit(); db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(body: UserLogin, db: Session = Depends(get_db)):
    # Accept username OR email in the username field
    identifier = body.username.strip()
    user = (db.query(User).filter(User.username == identifier).first()
            or db.query(User).filter(User.email == identifier).first())
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid username/email or password")
    return {"access_token": create_access_token(user.id), "token_type": "bearer"}


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UserOut)
def update_me(body: UserUpdate, db: Session = Depends(get_db),
              current_user: User = Depends(get_current_user)):
    if body.username and body.username != current_user.username:
        if db.query(User).filter(User.username == body.username).first():
            raise HTTPException(400, "Username already taken")
        current_user.username = body.username
    if body.email and body.email != current_user.email:
        if db.query(User).filter(User.email == body.email).first():
            raise HTTPException(400, "Email already registered")
        current_user.email = body.email
    db.commit(); db.refresh(current_user)
    return current_user


@router.post("/forgot-password")
def forgot_password(body: ForgotPassword, request: Request,
                    background_tasks: BackgroundTasks,
                    db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    # Always return ok — don't reveal whether email exists
    if user:
        token      = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=1)
        db.add(PasswordReset(user_id=user.id, token=token, expires_at=expires_at))
        db.commit()

        base_url  = str(request.base_url).rstrip("/")
        reset_url = f"{base_url}/?reset={token}"
        html = f"""
<div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:2rem;">
  <h2 style="color:#b96b30;">Stolcade Arena</h2>
  <p>Someone requested a password reset for your account.</p>
  <p style="margin:1.5rem 0;">
    <a href="{reset_url}"
       style="background:#b96b30;color:#fff;padding:0.75rem 1.5rem;
              border-radius:8px;text-decoration:none;font-weight:700;">
      Reset my password
    </a>
  </p>
  <p style="color:#888;font-size:0.85rem;">
    This link expires in 1 hour. If you didn't request this, you can ignore this email.
  </p>
</div>"""
        background_tasks.add_task(_send_email, user.email,
                                  "Reset your Stolcade password", html)
    return {"ok": True}


@router.post("/reset-password")
def reset_password(body: ResetPassword, db: Session = Depends(get_db)):
    reset = (db.query(PasswordReset)
               .filter(PasswordReset.token    == body.token,
                       PasswordReset.used     == False,
                       PasswordReset.expires_at > datetime.utcnow())
               .first())
    if not reset:
        raise HTTPException(400, "Invalid or expired reset link. Please request a new one.")
    if len(body.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    user = db.query(User).filter(User.id == reset.user_id).first()
    user.hashed_password = hash_password(body.new_password)
    reset.used = True
    db.commit()
    return {"ok": True}
