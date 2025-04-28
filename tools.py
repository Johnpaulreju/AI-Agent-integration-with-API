from langchain.tools import Tool
import httpx
import json

def user(mobile_number: str) -> str:
    """
    Fetch user details from the Odyssey API using the mobile number.
    
    Args:
        mobile_number (str): The mobile number to query.
    
    Returns:
        str: A formatted string listing all users with their username, patient ID, and email.
    """
    if not mobile_number.isdigit() or len(mobile_number) < 7:
        return "Invalid mobile number"

    url = ""
    headers = {
        "Content-Type": "application/json",
        "": ""
    }

    try:
        with httpx.Client() as client:
            response = client.get(
                f"{url}?mobile_number={mobile_number}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list):
                return "Error: Unexpected response format from API"
            
            if not data:
                return f"No users found with mobile number {mobile_number}"
            
            result = []
            for i, user_data in enumerate(data, 1):
                username = user_data.get("username", "Unknown")
                patient_id = user_data.get("patientId", "Unknown")
                email = user_data.get("email", "Unknown")
                result.append(f"user {i}: {username}, {patient_id} and {email}")
            
            return "\n".join(result)
                
    except httpx.HTTPStatusError as e:
        return f"Error: API request failed with status {e.response.status_code}"
    except httpx.RequestError as e:
        return f"Error: Network issue while contacting API - {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid response format from API"
    except Exception as e:
        return f"Error: Unexpected issue - {str(e)}"

def hospitals() -> str:
    """
    Fetch hospital details from the Odyssey API.
    
    Args:
        None
    
    Returns:
        str: A formatted string listing all hospitals with their name, ID, and site code, grouped by city.
    """
    url = "https://odyssey-concierge-api-v19.odysseypi.xyz/v1/hospitals"
    headers = {
        "Content-Type": "application/json",
        "odyssey-api-key": "0ed11aa234f59910d8c826adbc7afdad"
    }

    try:
        with httpx.Client() as client:
            response = client.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list):
                return "Error: Unexpected response format from API"
            
            if not data:
                return "No hospitals found"
            
            result = []
            hospital_index = 1
            for city_data in data:
                city_name = city_data.get("cityName", "Unknown")
                hospitals = city_data.get("hospitals", [])
                
                if not isinstance(hospitals, list):
                    continue
                
                for hospital_data in hospitals:
                    hospital_name = hospital_data.get("hospitalName", "Unknown")
                    hospital_id = hospital_data.get("hospitalId", "Unknown")
                    site_code = hospital_data.get("siteCode", "Unknown")
                    result.append(f"hospital {hospital_index}: {hospital_name}, {hospital_id} and {site_code} (City: {city_name})")
                    hospital_index += 1
            
            if not result:
                return "No hospitals found"
            
            return "\n".join(result)
                
    except httpx.HTTPStatusError as e:
        return f"Error: API request failed with status {e.response.status_code}"
    except httpx.RequestError as e:
        return f"Error: Network issue while contacting API - {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid response format from API"
    except Exception as e:
        return f"Error: Unexpected issue - {str(e)}"

# Define LangChain Tools
user_tool = Tool(
    name="user",
    description="Get user details (username, patient ID, and email) by mobile number using the Odyssey API",
    func=user
)

hospital_tool = Tool(
    name="hospitals",
    description="Get hospital details (name, ID, and site code) grouped by city using the Odyssey API",
    func=hospitals
)