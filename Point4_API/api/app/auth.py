from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os

security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if (
        credentials.username != os.getenv("API_USER")
        or credentials.password != os.getenv("API_PASSWORD")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    return True
