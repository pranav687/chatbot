# This file is for additional endpoints if needed
# The main endpoints are defined in main.py
# You can extend the API by adding more specialized endpoints here

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

from utils.config import CONCORDIA_CS_PROGRAM_INFO
from knowledge.concordia_data import ConcordiaDataClient

# Initialize router
router = APIRouter(prefix="/info", tags=["Information"])

# Initialize Concordia data client
concordia_data = ConcordiaDataClient()

# Pydantic models
class ProgramInfoResponse(BaseModel):
    program_info: Dict[str, Any]

class AdmissionRequirementsResponse(BaseModel):
    requirements: Dict[str, Any]

class DeadlinesResponse(BaseModel):
    deadlines: Dict[str, str]

class TuitionResponse(BaseModel):
    tuition: Dict[str, str]


@router.get("/program", response_model=ProgramInfoResponse)
async def get_program_info():
    """Get general program information"""
    try:
        return {"program_info": concordia_data.get_program_info()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving program info: {str(e)}")

@router.get("/admission", response_model=AdmissionRequirementsResponse)
async def get_admission_requirements(level: Optional[str] = None):
    """Get admission requirements by level (undergraduate/graduate)"""
    try:
        requirements = concordia_data.get_admission_requirements(level)
        return {"requirements": requirements}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving admission requirements: {str(e)}")

@router.get("/deadlines", response_model=DeadlinesResponse)
async def get_deadlines():
    """Get application deadlines"""
    try:
        deadlines = concordia_data.get_application_deadlines()
        return {"deadlines": deadlines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving deadlines: {str(e)}")

@router.get("/tuition", response_model=TuitionResponse)
async def get_tuition():
    """Get tuition information"""
    try:
        tuition = concordia_data.get_tuition_fees()
        return {"tuition": tuition}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving tuition info: {str(e)}")