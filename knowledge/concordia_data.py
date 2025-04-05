from typing import Dict, List, Any

from utils.config import CONCORDIA_CS_PROGRAM_INFO


class ConcordiaDataClient:
    """Client for accessing Concordia University data"""
    
    def __init__(self):
        """Initialize with program information"""
        self.program_info = CONCORDIA_CS_PROGRAM_INFO
    
    def get_program_info(self) -> Dict[str, Any]:
        """
        Get Computer Science program information
        
        Returns:
            Dictionary with program details
        """
        return self.program_info
    
    def get_admission_requirements(self, level: str = None) -> Dict[str, Any]:
        """
        Get admission requirements, optionally filtered by level
        
        Args:
            level: 'undergraduate' or 'graduate' or None for all
            
        Returns:
            Dictionary with admission requirements
        """
        if level and level.lower() in ["undergraduate", "graduate"]:
            return self.program_info["admission_requirements"][level.lower()]
        else:
            return self.program_info["admission_requirements"]
    
    def get_application_deadlines(self) -> Dict[str, str]:
        """
        Get application deadlines by term
        
        Returns:
            Dictionary with deadlines by term
        """
        return self.program_info["application_deadlines"]
    
    def get_tuition_fees(self) -> Dict[str, str]:
        """
        Get tuition fees by residency status
        
        Returns:
            Dictionary with tuition fees
        """
        return self.program_info["tuition_fees"]
    
    def format_program_summary(self) -> str:
        """
        Get a formatted summary of the program
        
        Returns:
            Formatted string with program summary
        """
        info = self.program_info
        
        summary = f"""
        # {info['name']} Program at Concordia University
        
        ## Overview
        The {info['name']} program is offered by the {', '.join(info['departments'])} 
        within the {info['faculty']} at Concordia University.
        
        ## Degrees Offered
        {', '.join(info['degrees'])}
        
        ## Admission Requirements
        
        ### Undergraduate
        - High School: {info['admission_requirements']['undergraduate']['high_school']}
        - CEGEP: {info['admission_requirements']['undergraduate']['cegep']}
        - Minimum GPA: {info['admission_requirements']['undergraduate']['minimum_gpa']}
        
        ### Graduate
        - Master's: {info['admission_requirements']['graduate']['masters']}
        - PhD: {info['admission_requirements']['graduate']['phd']}
        
        ## Application Deadlines
        - Fall Term: {info['application_deadlines']['fall']}
        - Winter Term: {info['application_deadlines']['winter']}
        - Summer Term: {info['application_deadlines']['summer']}
        
        ## Tuition Fees
        - Quebec Residents: {info['tuition_fees']['quebec_residents']}
        - Canadian Non-Quebec Residents: {info['tuition_fees']['canadian_non_quebec']}
        - International Students: {info['tuition_fees']['international']}
        
        For more information, visit: {info['program_website']}
        """
        
        return summary