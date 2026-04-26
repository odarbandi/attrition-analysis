import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)
from src.load_data import clean_employee_data


@pytest.fixture
def base_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "age": [30, 35, 40, 25, 28, 45],
            "monthly_income": [4000, 8000, 5000, 9000, 4500, 8500],
            "job_satisfaction": [1, 3, 2, 4, 1, 3],
            "overtime": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "travel_frequency": ["Frequent", "Rarely", "Frequent", "Rarely", "Frequent", "Rarely"],
            "years_at_company": [2, 8, 1, 10, 2, 7],
            "attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_zero_attrition():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "attrition": ["No", "No", "No"],
        }
    )
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leave():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "attrition": ["Yes", "Yes", "Yes"],
        }
    )
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(base_df):
    result = attrition_by_department(base_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_values(base_df):
    result = attrition_by_department(base_df)
    # Each department has 2 employees, 1 leaver → 50% each
    assert set(result["department"]) == {"Sales", "HR", "IT"}
    assert (result["employees"] == 2).all()
    assert (result["leavers"] == 1).all()
    assert (result["attrition_rate"] == 50.0).all()


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "department": ["Sales", "Sales", "HR", "HR", "HR"],
            "attrition": ["Yes", "Yes", "Yes", "No", "No"],
        }
    )
    result = attrition_by_department(df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(base_df):
    result = attrition_by_overtime(base_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_values():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "overtime": ["Yes", "Yes", "Yes", "No", "No", "No"],
            "attrition": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )
    result = attrition_by_overtime(df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["attrition_rate"] == 0.0


def test_attrition_by_overtime_partial():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "No", "No"],
            "attrition": ["Yes", "No", "Yes", "No"],
        }
    )
    result = attrition_by_overtime(df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["attrition_rate"] == 50.0
    assert no_row["attrition_rate"] == 50.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(base_df):
    result = average_income_by_attrition(base_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "monthly_income": [4000, 6000, 8000, 10000],
            "attrition": ["Yes", "Yes", "No", "No"],
        }
    )
    result = average_income_by_attrition(df)
    yes_avg = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    no_avg = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert yes_avg == 5000.0
    assert no_avg == 9000.0


def test_average_income_leavers_earn_less(base_df):
    result = average_income_by_attrition(base_df)
    yes_avg = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    no_avg = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert yes_avg < no_avg


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(base_df):
    result = satisfaction_summary(base_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rate_is_per_group_not_total_leavers():
    # 2 employees at satisfaction 1, both leave → 100%
    # 4 employees at satisfaction 3, none leave → 0%
    # If the old bug were present, sat=1 rate would be 2/2*100=100 (coincidentally correct)
    # but sat=3 rate would be 0/2*100=0 (also correct by accident)
    # Use a case that distinguishes: sat=2 has 1 leaver out of 2 employees → should be 50%
    # Old bug: 1 / total_leavers * 100 = 1/3*100 = 33.33%
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "job_satisfaction": [1, 1, 2, 2, 3, 3],
            "attrition": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    sat2 = result[result["job_satisfaction"] == 2].iloc[0]
    assert sat2["attrition_rate"] == 50.0, (
        "attrition_rate must be leavers/group_employees, not leavers/total_leavers"
    )


def test_satisfaction_summary_sorted_ascending(base_df):
    result = satisfaction_summary(base_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)


def test_satisfaction_summary_100_percent_group():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [1, 1, 4, 4],
            "attrition": ["Yes", "Yes", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    assert result[result["job_satisfaction"] == 1]["attrition_rate"].iloc[0] == 100.0
    assert result[result["job_satisfaction"] == 4]["attrition_rate"].iloc[0] == 0.0


# --- clean_employee_data ---

def test_clean_employee_data_raises_on_missing_columns():
    df = pd.DataFrame({"employee_id": [1], "department": ["Sales"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_employee_data(df)


def test_clean_employee_data_fills_missing_overtime():
    df = pd.DataFrame(
        {
            "employee_id": [1],
            "department": ["Sales"],
            "age": [30],
            "monthly_income": [5000],
            "job_satisfaction": [3],
            "overtime": [None],
            "travel_frequency": ["Rarely"],
            "years_at_company": [3],
            "attrition": ["No"],
        }
    )
    result = clean_employee_data(df)
    assert result["overtime"].iloc[0] == "No"


def test_clean_employee_data_fills_missing_income_with_median():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "department": ["Sales", "HR", "IT"],
            "age": [30, 35, 40],
            "monthly_income": [4000.0, None, 6000.0],
            "job_satisfaction": [3, 3, 3],
            "overtime": ["No", "No", "No"],
            "travel_frequency": ["Rarely", "Rarely", "Rarely"],
            "years_at_company": [2, 5, 8],
            "attrition": ["No", "No", "No"],
        }
    )
    result = clean_employee_data(df)
    assert result["monthly_income"].iloc[1] == 5000.0


def test_clean_employee_data_strips_whitespace():
    df = pd.DataFrame(
        {
            "employee_id": [1],
            "department": ["  Sales  "],
            "age": [30],
            "monthly_income": [5000],
            "job_satisfaction": [3],
            "overtime": [" Yes "],
            "travel_frequency": [" Frequent "],
            "years_at_company": [2],
            "attrition": ["Yes"],
        }
    )
    result = clean_employee_data(df)
    assert result["department"].iloc[0] == "Sales"
    assert result["overtime"].iloc[0] == "Yes"
