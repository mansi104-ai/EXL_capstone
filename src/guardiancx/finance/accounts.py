"""Synthetic customer accounts — the ledger the conversation is actually about.

Without this, the system could reason about a customer's *situation* but knew
nothing about their *money*: it could offer to restructure an instalment without
knowing what the instalment was. This module supplies that half — customers,
their accounts, balances, overdue instalments and recent transactions — so a call
can be about AED 8,650 rather than about "your instalment". Everything here is
fictional.

It also carries the relationships between customers, and that is the point of the
harder case in this file: **Meera and Rajesh Deshpande hold a joint home loan, and
Rajesh has a sole credit card.** A caller entitled to the joint account is not
thereby entitled to the sole one, and being his widow does not change that.
Bereavement is exactly when banks leak third-party data, because the request is
sympathetic and the caller is grieving. The ledger therefore distinguishes:

* **own** — the customer's sole accounts;
* **joint** — accounts held with someone else, which they may see in full;
* **third-party** — someone else's, which they may not see at all, whatever the
  relationship and whatever is said on the call.

Redaction happens at lookup, not by asking callers to remember:
:func:`disclosable_view` is the only way account data leaves this module, and it
masks IBANs and Emirates ID numbers even for the customer's own accounts. A
handler confirms an account by its last four digits.

**Language is a ledger field, not a call-time guess.** Every customer here has a
preferred language on file, because an outbound collections call has to choose a
voice and a set of approved wording *before* the customer has said anything. The
call can still switch when the customer answers in something else — but the
default comes from the record, the way it would in a real bank.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .taxonomy import Product

BANK_NAME = "Gulf Union Bank"
BANK_SHORT = "GUB"


def aed(amount: float) -> str:
    """Format in dirhams: AED 1,234,567.

    Fils are dropped on whole amounts. This is read aloud, and "eight thousand
    six hundred and fifty point zero zero dirhams" is not how anyone speaks.
    """
    negative = amount < 0
    whole, _, frac = f"{abs(amount):.2f}".partition(".")
    whole = f"{int(whole):,}"
    suffix = "" if frac == "00" else f".{frac}"
    return f"{'-' if negative else ''}AED {whole}{suffix}"


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
    iban: str
    account_number: str
    balance: float = 0.0            # negative on credit products = owed
    credit_limit: float = 0.0
    monthly_payment: float = 0.0    # the instalment, or the minimum due on a card
    next_payment_date: str = ""
    arrears_months: int = 0
    arrears_amount: float = 0.0
    salary_transfer: bool = False   # repayment is deducted from a salary credit
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
    def masked_iban(self) -> str:
        return self.iban[:6] + "*" * 13 + self.iban[-4:]

    @property
    def owed(self) -> float:
        """What the customer owes, as a positive number (0 for credit balances)."""
        return abs(self.balance) if self.balance < 0 else 0.0

    def summary(self) -> str:
        bits = [f"{self.label} ({self.masked_number})"]
        if self.owed:
            bits.append(f"{aed(self.owed)} outstanding")
        else:
            bits.append(f"{aed(self.balance)} available")
        if self.monthly_payment:
            bits.append(f"{aed(self.monthly_payment)} due {self.next_payment_date}")
        if self.arrears_months:
            bits.append(f"{self.arrears_months} instalment(s) overdue "
                        f"({aed(self.arrears_amount)})")
        return " · ".join(bits)


class Customer(BaseModel):
    customer_id: str
    name: str
    date_of_birth: str
    emirates_id: str                 # 784-YYYY-NNNNNNN-C
    emirate: str = ""
    mobile: str = ""
    email: str = ""
    nationality: str = ""
    language: str = "en"             # BCP-47 primary subtag; the call's default
    employer: str = ""               # on file, and never to be contacted about a debt
    vulnerability_note: str = ""     # a flag already on file from a previous call
    related: dict[str, str] = Field(default_factory=dict)  # customer_id -> relationship

    @property
    def first_name(self) -> str:
        return self.name.split()[0]

    @property
    def emirates_id_last4(self) -> str:
        digits = re.sub(r"\D", "", self.emirates_id)
        return digits[-4:] if len(digits) >= 4 else ""


def _recent(days_ago: int) -> str:
    return (date(2026, 8, 19) - timedelta(days=days_ago)).isoformat()


# --------------------------------------------------------------------------- #
# The book
#
# Six residents of the UAE, each exercising a different journey end to end, and
# one household that puts a third party's data within reach of a sympathetic
# request. The language spread is the point as much as the journeys are: a
# collections book in this market is not an English-speaking book.
# --------------------------------------------------------------------------- #
CUSTOMERS: dict[str, Customer] = {
    "CUST-4471": Customer(
        customer_id="CUST-4471", name="Meera Deshpande", date_of_birth="1958-03-12",
        emirates_id="784-1958-4417102-3", emirate="Dubai", mobile="+971 50 441 7102",
        email="meera.deshpande@example.ae", nationality="Indian", language="hi",
        employer="Retired",
        vulnerability_note="Bereavement recorded 14 July 2026 — single point of contact.",
        related={"CUST-4472": "spouse (deceased)"},
    ),
    "CUST-4472": Customer(
        customer_id="CUST-4472", name="Rajesh Deshpande", date_of_birth="1955-11-02",
        emirates_id="784-1955-4417203-9", emirate="Dubai", mobile="+971 50 441 7203",
        email="rajesh.deshpande@example.ae", nationality="Indian", language="hi",
        vulnerability_note="Deceased — succession documents outstanding.",
        related={"CUST-4471": "spouse"},
    ),
    "CUST-8830": Customer(
        customer_id="CUST-8830", name="Suresh Iyer", date_of_birth="1961-06-21",
        emirates_id="784-1961-8830194-6", emirate="Sharjah", mobile="+971 55 883 0194",
        email="s.iyer@example.ae", nationality="Indian", language="ml",
        employer="Al Nahda Trading LLC",
        vulnerability_note="Memory difficulties following a stroke; daughter assists.",
    ),
    "CUST-6612": Customer(
        # The collections case the whole outbound channel exists for: an
        # expatriate borrower on a salary-transfer loan whose employment — and
        # therefore whose residency — has just ended.
        customer_id="CUST-6612", name="Muhammad Aslam", date_of_birth="1988-01-30",
        emirates_id="784-1988-6612004-1", emirate="Sharjah", mobile="+971 52 661 2004",
        email="m.aslam@example.ae", nationality="Pakistani", language="ur",
        employer="Gulf Steel Fabrication LLC",
        vulnerability_note="",
    ),
    "CUST-7788": Customer(
        customer_id="CUST-7788", name="Fatima Al Suwaidi", date_of_birth="1977-09-08",
        emirates_id="784-1977-7788025-4", emirate="Abu Dhabi", mobile="+971 50 778 8025",
        email="f.alsuwaidi@example.ae", nationality="Emirati", language="ar",
        employer="Abu Dhabi Department of Education",
        vulnerability_note="Serious illness disclosed 2 August 2026.",
    ),
    "CUST-2205": Customer(
        customer_id="CUST-2205", name="Maria Santos", date_of_birth="1994-04-17",
        emirates_id="784-1994-2205031-2", emirate="Dubai", mobile="+971 56 220 5031",
        email="m.santos@example.ae", nationality="Filipino", language="fil",
        employer="Jumeirah Facilities Management",
    ),
}

ACCOUNTS: list[Account] = [
    # --- Meera & Rajesh Deshpande: joint home loan, his sole card -----------
    Account(
        account_id="ACC-1001", product=Product.MORTGAGE,
        label="Joint home loan", iban="AE070990000044710021884",
        account_number="44710021884",
        balance=-1_420_000.00, monthly_payment=8_650.00, next_payment_date="5 September",
        arrears_months=3, arrears_amount=25_950.00,
        holders=["CUST-4471", "CUST-4472"],
        transactions=[
            Transaction(date=_recent(4), description="Instalment — direct debit returned unpaid",
                        amount=0.0, category="arrears"),
            Transaction(date=_recent(35), description="Instalment — direct debit returned unpaid",
                        amount=0.0, category="arrears"),
        ],
    ),
    Account(
        account_id="ACC-1002", product=Product.SAVINGS,
        label="Joint savings account", iban="AE070990000044710038512",
        account_number="44710038512",
        balance=9_120.00, holders=["CUST-4471", "CUST-4472"],
        transactions=[
            Transaction(date=_recent(2), description="Pension credit", amount=7_100.00,
                        category="income"),
            Transaction(date=_recent(3), description="DEWA", amount=-1_090.00,
                        category="essential"),
            Transaction(date=_recent(6), description="Funeral and repatriation costs",
                        amount=-23_000.00, category="one-off"),
        ],
    ),
    Account(
        # Rajesh's sole card. Meera is not a holder — the guardrail case.
        account_id="ACC-1003", product=Product.CREDIT_CARD,
        label="Credit card", iban="AE070990000044720117340",
        account_number="44720117340",
        balance=-56_200.00, credit_limit=100_000.00, monthly_payment=2_810.00,
        next_payment_date="12 September", holders=["CUST-4472"],
    ),

    # --- Suresh Iyer ---------------------------------------------------------
    Account(
        account_id="ACC-2001", product=Product.SAVINGS,
        label="Savings account", iban="AE070990000088300194627",
        account_number="88300194627",
        balance=48_240.00, holders=["CUST-8830"],
        transactions=[
            Transaction(date=_recent(1), description="Salary credit", amount=16_200.00,
                        category="income"),
            Transaction(date=_recent(5), description="SEWA", amount=-480.00,
                        category="essential"),
        ],
    ),

    # --- Muhammad Aslam: job loss, salary-transfer loan + card ---------------
    Account(
        account_id="ACC-3001", product=Product.PERSONAL_LOAN,
        label="Personal loan", iban="AE070990000066120043159",
        account_number="66120043159",
        balance=-96_400.00, monthly_payment=4_130.00, next_payment_date="28 August",
        arrears_months=1, arrears_amount=4_130.00, salary_transfer=True,
        holders=["CUST-6612"],
        transactions=[
            Transaction(date=_recent(3), description="Instalment — no salary credit received",
                        amount=0.0, category="arrears"),
        ],
    ),
    Account(
        account_id="ACC-3002", product=Product.CREDIT_CARD,
        label="Credit card", iban="AE070990000066120118773",
        account_number="66120118773",
        balance=-18_650.00, credit_limit=20_000.00, monthly_payment=932.00,
        next_payment_date="9 September", holders=["CUST-6612"],
        transactions=[
            Transaction(date=_recent(2), description="Lulu Hypermarket groceries",
                        amount=-318.00, category="essential"),
            Transaction(date=_recent(9), description="Salary — final settlement and gratuity",
                        amount=21_400.00, category="income"),
        ],
    ),

    # --- Fatima Al Suwaidi ---------------------------------------------------
    Account(
        account_id="ACC-4001", product=Product.SAVINGS,
        label="Term deposit", iban="AE070990000077880250416",
        account_number="77880250416",
        balance=170_000.00, holders=["CUST-7788"],
    ),
    Account(
        account_id="ACC-4002", product=Product.PERSONAL_LOAN,
        label="Personal loan", iban="AE070990000077880266308",
        account_number="77880266308",
        balance=-134_000.00, monthly_payment=4_470.00, next_payment_date="3 September",
        salary_transfer=True, holders=["CUST-7788"],
    ),

    # --- Maria Santos --------------------------------------------------------
    Account(
        account_id="ACC-5001", product=Product.CREDIT_CARD,
        label="Credit card", iban="AE070990000022050311925",
        account_number="22050311925",
        balance=-12_380.00, credit_limit=15_000.00, monthly_payment=620.00,
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
    customer = get_customer(customer_id)
    related = set(customer.related) if customer else set()
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
            "iban": account.masked_iban,
            "balance": round(account.balance, 2),
            "owed": round(account.owed, 2),
            "monthly_payment": round(account.monthly_payment, 2),
            "next_payment_date": account.next_payment_date,
            "arrears_months": account.arrears_months,
            "arrears_amount": round(account.arrears_amount, 2),
            "salary_transfer": account.salary_transfer,
            "joint": account.is_joint,
            "recent_transactions": [t.model_dump() for t in account.transactions[:4]],
        })
    return {
        "customer": {
            "name": customer.name,
            "customer_id": customer.customer_id,
            "emirate": customer.emirate,
            "language": customer.language,
            "note_on_file": customer.vulnerability_note,
        },
        "accounts": accounts,
    }


def context_summary(customer_id: str) -> str:
    """A compact briefing for the agents — masked, and safe to put in a prompt."""
    view = disclosable_view(customer_id)
    if not view["customer"]:
        return "No account on file for this caller."
    lines = [f"Caller: {view['customer']['name']} ({customer_id}), "
             f"{view['customer']['emirate']}"]
    if view["customer"]["note_on_file"]:
        lines.append(f"Note already on file: {view['customer']['note_on_file']}")
    for account in view["accounts"]:
        parts = [f"{account['label']} {account['reference']}"]
        if account["owed"]:
            parts.append(f"{aed(account['owed'])} outstanding")
        else:
            parts.append(f"{aed(account['balance'])} available")
        if account["monthly_payment"]:
            parts.append(f"{aed(account['monthly_payment'])} due "
                         f"{account['next_payment_date']}")
        if account["arrears_months"]:
            parts.append(f"{account['arrears_months']} instalment(s) overdue "
                         f"({aed(account['arrears_amount'])})")
        if account["salary_transfer"]:
            parts.append("repaid by salary transfer")
        if account["joint"]:
            parts.append("joint account")
        lines.append("- " + " · ".join(parts))
    return "\n".join(lines)


def all_identifiers() -> set[str]:
    """Every full identifier in the book.

    The disclosure guardrail checks drafted replies against this: a value that
    appears here and was not masked has been read out of the ledger, whoever it
    belongs to.
    """
    values: set[str] = set()
    for account in ACCOUNTS:
        values.add(account.account_number)
        values.add(account.iban)
    for customer in CUSTOMERS.values():
        values.add(customer.mobile.replace(" ", ""))
        values.add(customer.email)
        values.add(customer.date_of_birth)
        values.add(customer.emirates_id)
    return {v for v in values if v}
