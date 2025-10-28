from __future__ import annotations
import os, time, math, random, uuid, threading
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, confloat
from .deps import get_current_claims, get_current_user_id
import jwt  # PyJWT
# settings.py（既存 main.py に直書きでもOK）
import os

JWT_SECRET = os.getenv("JWT_SIGNING_KEY", "qc_secret_2025")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ALLOWED_ISS = os.getenv("ALLOWED_ISS", "https://tobenicelife.com")

# ----------------------------------------------------------------------------
# 環境変数 / 設定
# ----------------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SIGNING_KEY", "qc_secret_2025")
JWT_ALG = "HS256"
ALLOWED_ISS = os.getenv("ALLOWED_ISS", "https://tobenicelife.com")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "https://tobenicelife.com,https://qc-front-cme8.vercel.app"
).split(",") if o.strip()]
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))

app = FastAPI(title="QC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pydantic import BaseModel

class JWTPayload(BaseModel):
    iss: str
    sub: str
    exp: int
    iat: Optional[int] = None
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None

# --- モデル定義 ---
class GenerateIn(BaseModel):
    topic: str
    level: int
    count: int

class Problem(BaseModel):
    id: str
    title: str
    body: Optional[str] = None

class GenerateOut(BaseModel):
    problems: List[Problem]

class GradeIn(BaseModel):
    questionId: str
    answer: str

class GradeOut(BaseModel):
    correct: bool
    feedback: Optional[dict] = None

class ProgressOut(BaseModel):
    ok: bool


# --- エンドポイント ---
@app.post("/generate", response_model=GenerateOut)
def generate(
    req: GenerateIn,
    claims=Depends(get_current_claims),
    user_id: str = Depends(get_current_user_id),
):
    print("✅ Token OK:", claims)
    return GenerateOut(
        problems=[
            Problem(
                id="q-1",
                title=f"{req.topic} Lv{req.level} の問題 (user:{user_id})",
                body="サンプル問題本文"
            )
        ]
    )

@app.post("/grade", response_model=GradeOut)
def grade(
    req: GradeIn,
    claims=Depends(get_current_claims),
    user_id: str = Depends(get_current_user_id),
):
    correct = req.answer.strip() == "1.23"  # 仮の採点
    feedback = None
    if not correct:
        feedback = {"message": "もう一度チャレンジ", "expected": 1.23, "tolerance": 0.05}
    return GradeOut(correct=correct, feedback=feedback)

@app.post("/progress", response_model=ProgressOut)
def progress(
    req: GradeIn,
    claims=Depends(get_current_claims),
    user_id: str = Depends(get_current_user_id),
):
    # user_id と questionId を保存する処理を入れる
    return ProgressOut(ok=True)

# ----------------------------------------------------------------------------
# 認証ヘルパー（Bearer か ?jwt のどちらでもOKに統一）
# ----------------------------------------------------------------------------
def _extract_token(req: Request) -> Optional[str]:
    auth = req.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    # クエリでも受ける
    q = req.query_params.get("jwt")
    return q if q else None

def _decode_and_validate(token: str) -> JWTPayload:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    if ALLOWED_ISS and payload.get("iss") != ALLOWED_ISS:
        raise HTTPException(status_code=401, detail="Invalid issuer")

    # Pydantic モデルにバリデートして返す
    try:
        return JWTPayload(**payload)
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed token payload")

async def get_current_user(req: Request) -> JWTPayload:
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    return _decode_and_validate(token)


# ----------------------------------------------------------------------------
# ヘルスチェック
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

# ----------------------------------------------------------------------------
# ここで /practice を定義（後半のAPIと同じ app にぶら下げる）
# ----------------------------------------------------------------------------
@app.get("/practice")
def practice(user: JWTPayload = Depends(get_current_user)):
    return {"ok": True, "user": {"id": user.sub, "name": user.name}}

# ----------------------------------------------------------------------------
# レート制限（簡易実装）
# ----------------------------------------------------------------------------
_rate_lock = threading.Lock()
_rate_bucket: Dict[str, List[float]] = {}


def rate_limit(req: Request):
    ip = req.client.host if req.client else "unknown"
    now = time.time()
    window = 60.0
    with _rate_lock:
        bucket = _rate_bucket.setdefault(ip, [])
        # 現在のウィンドウに残っている呼び出しのみ保持
        bucket = [t for t in bucket if now - t < window]
        if len(bucket) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(status_code=429, detail="Too Many Requests")
        bucket.append(now)
        _rate_bucket[ip] = bucket

# ----------------------------------------------------------------------------
# データモデル
# ----------------------------------------------------------------------------
class ProblemType(str):
    MEAN = "mean"
    VARIANCE = "variance"
    CORRELATION = "correlation"
    REGRESSION = "simple_regression"
    P_CHART = "p_chart"  # 管理図（p管理図：不良率）


class GenerateRequest(BaseModel):
    type: str = Field(..., description=f"{ProblemType.MEAN}|{ProblemType.VARIANCE}|{ProblemType.CORRELATION}|{ProblemType.REGRESSION}|{ProblemType.P_CHART}")
    level: int = Field(1, ge=1, le=3, description="難易度 1-3")
    n: int = Field(8, ge=4, le=200, description="データ数 or サンプル数")


class GeneratedProblem(BaseModel):
    problem_id: str
    type: str
    question: str
    data: Dict[str, Any]
    tolerance: float


class GradeRequest(BaseModel):
    problem_id: str
    # 回答は数値 or 複合（回帰: 係数/切片）に対応
    answer_number: Optional[confloat(ge=-1e12, le=1e12)] = None
    answer_tuple: Optional[Tuple[float, float]] = None  # (slope, intercept)


class GradeResult(BaseModel):
    correct: bool
    expected: Any
    score: float
    feedback: str


class ProgressSummary(BaseModel):
    user_id: str
    total: int
    correct: int
    accuracy: float
    by_type: Dict[str, Dict[str, int]]  # {type: {total, correct}}

# ----------------------------------------------------------------------------
# メモリ内ストア（本番はDBへ）
# ----------------------------------------------------------------------------
_store_lock = threading.Lock()
_problems: Dict[str, Dict[str, Any]] = {}
_progress: Dict[str, Dict[str, Any]] = {}

# 問題保存構造：
# _problems[problem_id] = {
#   "type": str,
#   "question": str,
#   "data": {...},
#   "answer": Any,           # 数値 or (slope, intercept)
#   "tolerance": float,
#   "created_at": epoch,
#   "user_id": str
# }

# _progress[user_id] = {
#   "totals": {"total": int, "correct": int},
#   "by_type": {type: {"total": int, "correct": int}}
# }

# ----------------------------------------------------------------------------
# ユーティリティ（計算）
# ----------------------------------------------------------------------------

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def variance(xs: List[float], ddof: int = 0) -> float:
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - ddof)


def correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("length mismatch")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def simple_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    # y = a x + b （a: slope, b: intercept）
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        a = 0.0
    else:
        a = sxy / sxx
    b = my - a * mx
    return a, b

# ----------------------------------------------------------------------------
# 問題生成器
# ----------------------------------------------------------------------------

def gen_numeric_dataset(n: int, mu: float, sigma: float) -> List[float]:
    return [round(random.gauss(mu, sigma), 2) for _ in range(n)]


def gen_pairs_with_correlation(n: int, a: float, b: float, noise: float) -> Tuple[List[float], List[float]]:
    xs = [round(random.uniform(10, 90), 2) for _ in range(n)]
    ys = [round(a * x + b + random.gauss(0, noise), 2) for x in xs]
    return xs, ys


def build_problem(req: GenerateRequest, user_id: str) -> GeneratedProblem:
    level = req.level
    if req.type == ProblemType.MEAN:
        xs = gen_numeric_dataset(req.n, mu=50, sigma=10/level)
        ans = round(mean(xs), 3)
        tol = 0.5 / level
        q = f"次のデータの平均値を小数第3位まで求めよ（許容誤差±{tol}）: {xs}"
        data = {"xs": xs}
    elif req.type == ProblemType.VARIANCE:
        xs = gen_numeric_dataset(req.n, mu=60, sigma=12/level)
        # 分散は母分散（ddof=0）で統一
        ans = round(variance(xs, ddof=0), 3)
        tol = 1.0 / level
        q = f"次のデータの分散（母分散）を小数第3位まで求めよ（許容誤差±{tol}）: {xs}"
        data = {"xs": xs}
    elif req.type == ProblemType.CORRELATION:
        a = round(random.uniform(0.3, 1.2), 2)
        b = round(random.uniform(-10, 10), 1)
        noise = 8.0 / level
        xs, ys = gen_pairs_with_correlation(req.n, a, b, noise)
        ans = round(correlation(xs, ys), 3)
        tol = 0.05 / level
        q = "次のデータの相関係数 r を小数第3位まで求めよ（許容誤差±{tol}）".format(tol=tol)
        data = {"xs": xs, "ys": ys}
    elif req.type == ProblemType.REGRESSION:
        a_true = round(random.uniform(0.2, 1.5), 2)
        b_true = round(random.uniform(-20, 20), 1)
        noise = 10.0 / level
        xs, ys = gen_pairs_with_correlation(req.n, a_true, b_true, noise)
        a_hat, b_hat = simple_regression(xs, ys)
        ans = (round(a_hat, 3), round(b_hat, 3))
        tol = 0.1 / level
        q = "次のデータに対して最小二乗法による単回帰直線 y = a x + b を推定せよ（a, b を小数第3位まで。各許容誤差±{tol}）".format(tol=tol)
        data = {"xs": xs, "ys": ys}
    elif req.type == ProblemType.P_CHART:
        # p管理図：各サンプルのnは10〜100の可変、総数と不良数
        groups = []
        base_p = random.uniform(0.02, 0.12)
        for _ in range(req.n):
            n_i = random.randint(25, 120)
            defects = sum(1 for _ in range(n_i) if random.random() < base_p)
            groups.append({"n": n_i, "defects": defects})
        # 総計
        N = sum(g["n"] for g in groups)
        D = sum(g["defects"] for g in groups)
        pbar = D / N if N else 0.0
        ans = round(pbar, 4)
        tol = 0.01 / max(1, level)
        q = "次の検査データから p管理図の中心線 p̄ を小数第4位まで求めよ（許容誤差±{tol}）".format(tol=tol)
        data = {"groups": groups}
    else:
        raise HTTPException(status_code=400, detail="Unsupported problem type")

    problem_id = str(uuid.uuid4())
    with _store_lock:
        _problems[problem_id] = {
            "type": req.type,
            "question": q,
            "data": data,
            "answer": ans,
            "tolerance": tol,
            "created_at": time.time(),
            "user_id": user_id,
        }
    return GeneratedProblem(problem_id=problem_id, type=req.type, question=q, data=data, tolerance=tol)

# ----------------------------------------------------------------------------
# 採点
# ----------------------------------------------------------------------------

def is_within_tolerance(user_value: float, expected: float, tol: float) -> bool:
    return abs(user_value - expected) <= tol


def grade_answer(problem_id: str, user_answer: GradeRequest, user_id: str) -> GradeResult:
    with _store_lock:
        rec = _problems.get(problem_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Problem not found or expired")
    if rec["user_id"] != user_id:
        # 他人の問題IDを使った採点を防止
        raise HTTPException(status_code=403, detail="Not your problem")

    expected = rec["answer"]
    tol = rec["tolerance"]
    ptype = rec["type"]

    correct = False
    feedback = ""

    if isinstance(expected, tuple):
        if not user_answer.answer_tuple:
            raise HTTPException(status_code=400, detail="Expected tuple answer (a, b)")
        a_exp, b_exp = expected
        a_in, b_in = user_answer.answer_tuple
        ca = is_within_tolerance(a_in, a_exp, tol)
        cb = is_within_tolerance(b_in, b_exp, tol)
        correct = ca and cb
        feedback = f"a: 期待値 {a_exp} / あなた {a_in}、b: 期待値 {b_exp} / あなた {b_in}（許容±{tol}）"
    else:
        if user_answer.answer_number is None:
            raise HTTPException(status_code=400, detail="Expected numeric answer")
        correct = is_within_tolerance(float(user_answer.answer_number), float(expected), tol)
        feedback = f"期待値 {expected} / あなた {user_answer.answer_number}（許容±{tol}）"

    score = 1.0 if correct else 0.0

    # 進捗へ反映
    with _store_lock:
        prog = _progress.setdefault(user_id, {"totals": {"total": 0, "correct": 0}, "by_type": {}})
        prog["totals"]["total"] += 1
        if correct:
            prog["totals"]["correct"] += 1
        tp = prog["by_type"].setdefault(ptype, {"total": 0, "correct": 0})
        tp["total"] += 1
        if correct:
            tp["correct"] += 1

    return GradeResult(correct=correct, expected=expected, score=score, feedback=feedback)

# ----------------------------------------------------------------------------
# 進捗API
# ----------------------------------------------------------------------------

def get_progress(user_id: str) -> ProgressSummary:
    with _store_lock:
        prog = _progress.get(user_id, {"totals": {"total": 0, "correct": 0}, "by_type": {}})
        total = prog["totals"]["total"]
        correct = prog["totals"]["correct"]
        by_type = prog["by_type"]
    acc = (correct / total) if total > 0 else 0.0
    return ProgressSummary(user_id=user_id, total=total, correct=correct, accuracy=round(acc, 3), by_type=by_type)

# ----------------------------------------------------------------------------
# ルーター
# ----------------------------------------------------------------------------

# ===== フロント契約に合わせたAPI（/api/* のプロキシ先想定） =====

from typing import List, Optional
from pydantic import BaseModel

# フロント契約用DTO
class FrontGenerateIn(BaseModel):
    topic: str        # "mean" | "variance" | "correlation" | "pchart" | "regression"
    level: int        # 1..5 くらい想定（内部では1..3に丸め）
    count: int        # 1..20

class FrontGeneratedProblem(BaseModel):
    id: str
    title: str
    body: Optional[str] = None

class FrontGenerateOut(BaseModel):
    problems: List[FrontGeneratedProblem]

class FrontGradeIn(BaseModel):
    questionId: str
    answer: str       # 文字列で来る（数値 or "a,b" の可能性）

class FrontGradeOut(BaseModel):
    correct: bool
    feedback: Optional[dict] = None

class FrontProgressIn(BaseModel):
    questionId: str

class FrontProgressOut(BaseModel):
    ok: bool


def _map_topic_for_engine(front_topic: str) -> str:
    # フロントの "regression" はエンジン側 "simple_regression"
    if front_topic == "regression":
        return "simple_regression"
    if front_topic == "pchart":
        return "p_chart"
    return front_topic  # mean / variance / correlation はそのまま


@app.post("/generate", response_model=FrontGenerateOut)
def generate_endpoint(req: FrontGenerateIn, request: Request, user: JWTPayload = Depends(get_current_user)):
    rate_limit(request)
    # level は内部の 1..3 に丸める（上がってきたら3で頭打ち）
    lvl = max(1, min(3, int(req.level)))
    cnt = max(1, min(20, int(req.count)))

    problems: List[FrontGeneratedProblem] = []
    ptype = _map_topic_for_engine(req.topic)

    from pydantic import BaseModel, Field  # 既存の GenerateRequest を使う
    class _Gen(BaseModel):
        type: str
        level: int
        n: int

    # 1問 = n(データ数)はレベルに応じて適当に（必要ならUIから別指定でもOK）
    n_data = 12 if ptype in ("mean", "variance", "correlation", "simple_regression") else 8

    for _ in range(cnt):
        gp = build_problem(
            _Gen(type=ptype, level=lvl, n=n_data),
            user_id=user.sub
        )
        # フロントの期待形へ整形
        problems.append(FrontGeneratedProblem(
            id=gp.problem_id,
            title=gp.question,
            body=None  # 問題文に全て含める運用。必要なら補足を入れる
        ))

    return FrontGenerateOut(problems=problems)


@app.post("/grade", response_model=FrontGradeOut)
def grade_endpoint(req: FrontGradeIn, request: Request, user: JWTPayload = Depends(get_current_user)):
    rate_limit(request)

    # 文字列answerを数値 or (a,b) にパース
    ans_str = (req.answer or "").strip()
    from pydantic import BaseModel
    class _Grade(BaseModel):
        problem_id: str
        answer_number: Optional[float] = None
        answer_tuple: Optional[Tuple[float, float]] = None

    payload = _Grade(problem_id=req.questionId)

    if "," in ans_str:
        # "a,b" 形式 → 回帰の想定
        try:
            a_str, b_str = ans_str.split(",", 1)
            payload.answer_tuple = (float(a_str), float(b_str))
        except Exception:
            raise HTTPException(status_code=400, detail="Answer format for regression should be 'a,b'")
    else:
        # 単数値
        try:
            payload.answer_number = float(ans_str)
        except Exception:
            raise HTTPException(status_code=400, detail="Answer must be numeric or 'a,b'")

    r = grade_answer(payload.problem_id, payload, user_id=user.sub)

    fb = {"message": r.feedback, "expected": r.expected, "tolerance": None}
    return FrontGradeOut(correct=r.correct, feedback=fb)


@app.post("/progress", response_model=FrontProgressOut)
def progress_endpoint(req: FrontProgressIn, request: Request, user: JWTPayload = Depends(get_current_user)):
    # 今は「採点時に内部ストアに加算」→ ここではOKだけ返す
    rate_limit(request)
    _ = req.questionId  # 使わないが契約上受け取る
    return FrontProgressOut(ok=True)


# ----------------------------------------------------------------------------
# 起動: uvicorn main:app --host 0.0.0.0 --port 10000
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# README（運用ノート）
# ----------------------------------------------------------------------------
README = r"""
# QC API - Backend Feature Pack (v1)

## 環境変数
- JWT_SIGNING_KEY: 署名鍵（WPと一致）
- ALLOWED_ISS: https://tobenicelife.com
- ALLOWED_ORIGINS: カンマ区切り（例）https://tobenicelife.com,https://qc-front-cme8.vercel.app
- RATE_LIMIT_PER_MIN: 120（無料枠は控えめ推奨）

## 主要エンドポイント
- GET /health
- POST /generate
  - body: {"type":"mean|variance|correlation|simple_regression|p_chart","level":1..3,"n":データ数}
- POST /grade
  - body: {"problem_id":"...","answer_number":数値} または {"answer_tuple":[a,b]}
- GET /progress

## cURL例
TOKEN="$(php -r 'echo base64_encode(random_bytes(8));')"  # ダミー。WPのJWTに置換

curl -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"type":"mean","level":1,"n":8}' \
     https://<your-render-domain>/generate

## TODO/拡張案
- DB（PostgreSQL）で問題・履歴を永続化
- 問題テンプレの難易度制御（分布、外れ値、データ桁など）
- 学習分析（誤答パターン別の出題バイアス）
- JWTのaud（受信者）検証、JTIで再利用防止
- 管理図：Xbar-R、Xbar-S、c/u管理図、3σ線の算出
- 採点詳細：丸め規則や有効数字を厳密化
"""

