"""租户身份只从 Bearer key 推导，绝不读请求参数。"""
from fastapi import HTTPException, Request


def require_tenant(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少凭证")
    key = auth[len("Bearer "):].strip()
    tenant = request.app.state.config.tenant_by_key(key)
    if tenant is None:
        raise HTTPException(status_code=401, detail="凭证无效")
    return tenant.name
