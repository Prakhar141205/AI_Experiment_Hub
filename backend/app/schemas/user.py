from pydantic import BaseModel, ConfigDict, EmailStr, Field



class UserCreate(BaseModel):

    username: str = Field(..., max_length=255)
    email: EmailStr = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=255)


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: int
    username: str
    email: EmailStr
    account_status: bool


class UserLogin(BaseModel):
    username_or_email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=128)
