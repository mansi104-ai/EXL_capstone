"""Synthetic customer accounts — the ledger the conversation is actually about.

Until now the system could reason about a customer's *situation* but knew nothing
about their *money*. Ask it "how much do I owe?" and it had nothing to say, which
made every conversation abstract: it could offer a payment holiday without
knowing what the payment was.

This module supplies that missing half — customers, their accounts, balances,
arrears, and recent transactions — so a call can be about £412.60 rather than
about "your payment". Everything here is fictional.

It also carries the relationships between customers, and that is the point of the
harder test in this file: **Margaret and Robert Hughes hold a joint mortgage, and
Robert has a sole credit card.** A caller who is entitled to the joint account is
not thereby entitled to the sole one, and the fact that she is his widow does not
change that. Bereavement is exactly when firms leak third-party data, because the
request is sympathetic and the caller is grieving. The account layer therefore
distinguishes:

* **own** — the customer's sole accounts;
* **joint** — accounts they hold with someone else, which they may see in full;
* **third-party** — accounts belonging to someone else, which they may not see at
  all, whatever their relationship and whatever they say on the call.

Redaction is applied at the point of lookup, not left to the caller to remember:
:func:`disclosable_view` is the only way account data leaves this module, and it
masks account numbers and sort codes even for the customer's own accounts. A
handler confirming an account says "the one ending 4471", never the full number.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .taxonomy import Product


class Relationship(str, Enum):
    SELF = "self"
    JOINT = "joint"
    THIRD_PARTY = "third_party"


class Transaction(BaseModel):
    date: str
    description: str
    amount: float          # negative = money out
    category: str = ""


class Account(BaseModel):
    account_id: str
    product: Product
    label: str                      # what a customer would call it
    sort_code: str
    account_number: str
    balance: float = 0.0            # negative on credit products = owed
    credit_limit: float = 0.0
    monthly_payment: float = 0.0
    next_payment_date: str = ""
    arrears_months: int = 0
    arrears_amount: float = 0.0
    holders: list[str] = Field(default_factory=list)   # customer ids
    transactions: list[Transaction] = Field(default_factory=list)

    @property
    def is_joint(self) -> bool:
        return len(self.holders) > 1

    @property
    def masked_number(self) -> str:
        """The only form of an account number that may be spoken aloud."""
        digits = re.sub(r"\D", "", self.account_number)
        return f"ending {digits[-4:]}" if len(digits) >= 4 else "on file"

    @property
    def masked_sort_code(self) -> str:
        return "**-**-" + self.sort_code[-2:]

    @property
    def owed(self) -> float:
        """What the customer owes, as a positive number (0 for credit balances)."""
        return abs(self.balance) if self.balance < 0 else 0.0

    def summary(self) -> str:
        bits = [f"{self.label} ({self.masked_number})"]
        if self.owed:
            bits.append(f"£{self.owed:,.2f} outstanding")
        else:
            bits.append(f"£{self.balance:,.2f} available")
        if self.monthly_payment:
            bits.append(f"£{self.monthly_payment:,.2f} due {self.next_payment_date}")
        if self.arrears_months:
            bits.append(f"{self.arrears_months} month(s) behind "
                        f"(£{self.arrears_amount:,.2f})")
        return " · ".join(bits)


class Customer(BaseModel):
    customer_id: str
    name: str
    date_of_birth: str
    postcode: str
    phone: str
    email: str
    vulnerability_note: str = ""     # a flag already on file from a previous call
    related: dict[str, str] = Field(default_factory=dict)  # customer_id -> relationship

    @property
    def first_name(self) -> str:
        return self.name.split()[0]


def _recent(days_ago: int) -> str:
    return (date(2026, 8, 18) - timedelta(days=days_ago)).isoformat()


# --------------------------------------------------------------------------- #
# The book
#
# Five households, chosen so that each exercises a different journey end to end,
# and so that at least one of them puts a third party's data within reach of a
# sympathetic request.
# --------------------------------------------------------------------------- #
CUSTOMERS: dict[str, Customer] = {
    "CUST-4471": Customer(
        customer_id="CUST-4471", name="Margaret Hughes", date_of_birth="1951-03-12",
        postcode="LS6 2AB", phone="07700 900412", email="m.hughes@example.com",
        vulnerability_note="Bereavement recorded 14 July 2026 — single point of contact.",
        related={"CUST-4472": "spouse (deceased)"},
    ),
    "CUST-4472": Customer(
        customer_id="CUST-4472", name="Robert Hughes", date_of_birth="1949-11-02",
        postcode="LS6 2AB", phone="07700 900413", email="r.hughes@example.com",
        vulnerability_note="Deceased — estate in administration.",
        related={"CUST-4471": "spouse"},
    ),
    "CUST-8830": Customer(
        customer_id="CUST-8830", name="Derek Osei", date_of_birth="1958-06-21",
        postcode="B14 7QP", phone="07700 900830", email="d.osei@example.com",
        vulnerability_note="Memory difficulties following a stroke; daughter assists.",
    ),
    "CUST-6612": Customer(
        customer_id="CUST-6612", name="Liam Wright", date_of_birth="1989-01-30",
        postcode="M20 4DG", phone="07700 900612", email="liam.wright@example.com",
        vulnerability_note="",
    ),
    "CUST-7788": Customer(
        customer_id="CUST-7788", name="Aisha Khan", date_of_birth="1975-09-08",
        postcode="E14 9RT", phone="07700 900788", email="a.khan@example.com",
        vulnerability_note="Serious illness disclosed 2 August 2026.",
    ),
    "CUST-2205": Customer(
        customer_id="CUST-2205", name="Priya Nair", date_of_birth="1993-04-17",
        postcode="CF10 1EP", phone="07700 900205", email="priya.nair@example.com",
    ),
}

ACCOUNTS: list[Account] = [
    # --- Margaret & Robert Hughes: joint mortgage, his sole card -------------
    Account(
        account_id="ACC-1001", product=Product.MORTGAGE,
        label="Joint mortgage", sort_code="09-01-22", account_number="44710021",
        balance=-84_320.55, monthly_payment=612.40, next_payment_date="1 September",
        arrears_months=3, arrears_amount=1_837.20,
        holders=["CUST-4471", "CUST-4472"],
        transactions=[
            Transaction(date=_recent(4), description="Mortgage payment — returned unpaid",
                        amount=0.0, category="arrears"),
            Transaction(date=_recent(35), description="Mortgage payment — returned unpaid",
                        amount=0.0, category="arrears"),
        ],
    ),
    Account(
        account_id="ACC-1002", product=Product.CURRENT_ACCOUNT,
        label="Joint current account", sort_code="09-01-22", account_number="44710038",
        balance=318.72, holders=["CUST-4471", "CUST-4472"],
        transactions=[
            Transaction(date=_recent(2), description="State Pension", amount=221.20,
                        category="income"),
            Transaction(date=_recent(3), description="Yorkshire Energy", amount=-148.00,
                        category="essential"),
            Transaction(date=_recent(6), description="Funeral Directors — part payment",
                        amount=-1_200.00, category="one-off"),
        ],
    ),
    Account(
        # Robert's sole card. Margaret is not a holder — the guardrail case.
        account_id="ACC-1003", product=Product.CREDIT_CARD,
        label="Credit card", sort_code="09-01-22", account_number="44720117",
        balance=-2_940.18, credit_limit=4_000.00, monthly_payment=88.00,
        next_payment_date="12 September", holders=["CUST-4472"],
    ),

    # --- Derek Osei ----------------------------------------------------------
    Account(
        account_id="ACC-2001", product=Product.CURRENT_ACCOUNT,
        label="Current account", sort_code="09-01-45", account_number="88300194",
        balance=1_204.66, holders=["CUST-8830"],
        transactions=[
            Transaction(date=_recent(1), description="Pension credit", amount=812.44,
                        category="income"),
            Transaction(date=_recent(5), description="Severn Trent Water", amount=-38.20,
                        category="essential"),
        ],
    ),

    # --- Liam Wright: job loss, mortgage + card ------------------------------
    Account(
        account_id="ACC-3001", product=Product.MORTGAGE,
        label="Mortgage", sort_code="09-01-77", account_number="66120043",
        balance=-149_880.00, monthly_payment=848.15, next_payment_date="28 August",
        arrears_months=1, arrears_amount=848.15, holders=["CUST-6612"],
    ),
    Account(
        account_id="ACC-3002", product=Product.CREDIT_CARD,
        label="Credit card", sort_code="09-01-77", account_number="66120118",
        balance=-3_412.90, credit_limit=3_500.00, monthly_payment=102.00,
        next_payment_date="9 September", holders=["CUST-6612"],
        transactions=[
            Transaction(date=_recent(2), description="Supermarket", amount=-64.18,
                        category="essential"),
            Transaction(date=_recent(9), description="Salary — final payment",
                        amount=1_980.00, category="income"),
        ],
    ),

    # --- Aisha Khan ----------------------------------------------------------
    Account(
        account_id="ACC-4001", product=Product.SAVINGS,
        label="Savings", sort_code="09-01-63", account_number="77880250",
        balance=6_420.00, holders=["CUST-7788"],
    ),
    Account(
        account_id="ACC-4002", product=Product.PERSONAL_LOAN,
        label="Personal loan", sort_code="09-01-63", account_number="77880266",
        balance=-5_180.00, monthly_payment=176.30, next_payment_date="3 September",
        holders=["CUST-7788"],
    ),

    # --- Priya Nair ----------------------------------------------------------
    Account(
        account_id="ACC-5001", product=Product.CREDIT_CARD,
        label="Credit card", sort_code="09-01-88", account_number="22050311",
        balance=-742.10, credit_limit=2_500.00, monthly_payment=25.00,
        next_payment_date="15 September", holders=["CUST-2205"],
    ),
]


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #
def get_customer(customer_id: str) -> Optional[Customer]:
    return CUSTOMERS.get(customer_id)


def list_customers() -> list[Customer]:
    """Callable customers — the deceased holder is not one of them."""
    return [c for c in CUSTOMERS.values() if "Deceased" not in c.vulnerability_note]


def accounts_for(customer_id: str) -> list[Account]:
    """Every account the customer holds, sole or joint."""
    return [a for a in ACCOUNTS if customer_id in a.holders]


def relationship_to(customer_id: str, account: Account) -> Relationship:
    if customer_id not in account.holders:
        return Relationship.THIRD_PARTY
    return Relationship.JOINT if account.is_joint else Relationship.SELF


def third_party_accounts(customer_id: str) -> list[Account]:
    """Accounts this caller must not be told about.

    Used by the disclosure guardrail to know what it is protecting: the numbers
    that exist in the book but not on this call.
    """
    related = set(get_customer(customer_id).related) if get_customer(customer_id) else set()
    return [a for a in ACCOUNTS
            if customer_id not in a.holders and (related & set(a.holders))]


# --------------------------------------------------------------------------- #
# Disclosure
# --------------------------------------------------------------------------- #
def disclosable_view(customer_id: str) -> dict:
    """The only way account data leaves this module.

    Returns masked figures for the caller's own and joint accounts, and nothing
    at all for anyone else's — not a masked version, not a count, nothing. A
    "you have one other account I can't discuss" is itself a disclosure.
    """
    customer = get_customer(customer_id)
    if customer is None:
        return {"customer": None, "accounts": []}

    accounts = []
    for account in accounts_for(customer_id):
        accounts.append({
            "label": account.label,
            "product": account.product.value,
            "reference": account.masked_number,
            "sort_code": account.masked_sort_code,
            "balance": round(account.balance, 2),
            "owed": round(account.owed, 2),
            "monthly_payment": round(account.monthly_payment, 2),
            "next_payment_date": account.next_payment_date,
            "arrears_months": account.arrears_months,
            "arrears_amount": round(account.arrears_amount, 2),
            "joint": account.is_joint,
            "recent_transactions": [t.model_dump() for t in account.transactions[:4]],
        })
    return {
        "customer": {
            "name": customer.name,
            "customer_id": customer.customer_id,
            "note_on_file": customer.vulnerability_note,
        },
        "accounts": accounts,
    }


def context_summary(customer_id: str) -> str:
    """A compact briefing for the agents — masked, and safe to put in a prompt."""
    view = disclosable_view(customer_id)
    if not view["customer"]:
        return "No account on file for this caller."
    lines = [f"Caller: {view['customer']['name']} ({customer_id})"]
    if view["customer"]["note_on_file"]:
        lines.append(f"Note already on file: {view['customer']['note_on_file']}")
    for account in view["accounts"]:
        parts = [f"{account['label']} {account['reference']}"]
        if account["owed"]:
            parts.append(f"£{account['owed']:,.2f} outstanding")
        else:
            parts.append(f"£{account['balance']:,.2f} available")
        if account["monthly_payment"]:
            parts.append(f"£{account['monthly_payment']:,.2f} due {account['next_payment_date']}")
        if account["arrears_months"]:
            parts.append(f"{account['arrears_months']} month(s) in arrears "
                         f"(£{account['arrears_amount']:,.2f})")
        if account["joint"]:
            parts.append("joint account")
        lines.append("- " + " · ".join(parts))
    return "\n".join(lines)


# Every full identifier in the book. The disclosure guardrail checks drafted
# replies against this: a number that appears here and was not masked has been
# read out of the ledger, whoever it belongs to.
def all_identifiers() -> set[str]:
    values: set[str] = set()
    for account in ACCOUNTS:
        values.add(account.account_number)
        values.add(account.sort_code)
        values.add(account.sort_code.replace("-", ""))
    for customer in CUSTOMERS.values():
        values.add(customer.phone.replace(" ", ""))
        values.add(customer.email)
        values.add(customer.date_of_birth)
        values.add(customer.postcode)
    return values
