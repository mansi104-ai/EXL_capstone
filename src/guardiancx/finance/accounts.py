"""Synthetic customer accounts — the ledger the conversation is actually about.

Without this, the system could reason about a customer's *situation* but knew
nothing about their *money*: it could offer to restructure an EMI without knowing
what the EMI was. This module supplies that half — customers, their accounts,
balances, overdue EMIs and recent transactions — so a call can be about
₹24,850 rather than about "your instalment". Everything here is fictional.

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
masks account numbers and IFSC codes even for the customer's own accounts. A
handler confirms an account by its last four digits.

Amounts are formatted the way an Indian customer reads them — ₹1,12,400, not
₹112,400 — because a figure a customer has to re-parse is a figure they do not
take in.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .taxonomy import Product

BANK_NAME = "Pan Indian Bank"
BANK_SHORT = "PIB"


def inr(amount: float) -> str:
    """Format in the Indian numbering system: ₹12,34,567.00.

    Grouping runs in twos after the last three digits, so 1234567 becomes
    12,34,567. Getting this wrong is a small thing that makes an Indian product
    read as a translated foreign one.
    """
    negative = amount < 0
    whole, _, frac = f"{abs(amount):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        whole = f"{head},{tail}"
    # Drop the paise on whole amounts. This is read aloud, and "seventy-four
    # thousand five hundred and fifty point zero zero" is not how anyone speaks.
    suffix = "" if frac == "00" else f".{frac}"
    return f"{'-' if negative else ''}₹{whole}{suffix}"


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
    ifsc: str
    account_number: str
    balance: float = 0.0            # negative on credit products = owed
    credit_limit: float = 0.0
    monthly_payment: float = 0.0    # the EMI, or the minimum due on a card
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
    def masked_ifsc(self) -> str:
        return self.ifsc[:4] + "****" + self.ifsc[-2:]

    @property
    def owed(self) -> float:
        """What the customer owes, as a positive number (0 for credit balances)."""
        return abs(self.balance) if self.balance < 0 else 0.0

    def summary(self) -> str:
        bits = [f"{self.label} ({self.masked_number})"]
        if self.owed:
            bits.append(f"{inr(self.owed)} outstanding")
        else:
            bits.append(f"{inr(self.balance)} available")
        if self.monthly_payment:
            bits.append(f"{inr(self.monthly_payment)} due {self.next_payment_date}")
        if self.arrears_months:
            bits.append(f"{self.arrears_months} instalment(s) overdue "
                        f"({inr(self.arrears_amount)})")
        return " · ".join(bits)


class Customer(BaseModel):
    customer_id: str
    name: str
    date_of_birth: str
    pin_code: str                    # 6-digit postal code
    city: str = ""
    mobile: str = ""
    email: str = ""
    pan: str = ""                    # permanent account number
    vulnerability_note: str = ""     # a flag already on file from a previous call
    related: dict[str, str] = Field(default_factory=dict)  # customer_id -> relationship

    @property
    def first_name(self) -> str:
        return self.name.split()[0]


def _recent(days_ago: int) -> str:
    return (date(2026, 8, 19) - timedelta(days=days_ago)).isoformat()


# --------------------------------------------------------------------------- #
# The book
#
# Five households, each exercising a different journey end to end, and one that
# puts a third party's data within reach of a sympathetic request.
# --------------------------------------------------------------------------- #
CUSTOMERS: dict[str, Customer] = {
    "CUST-4471": Customer(
        customer_id="CUST-4471", name="Meera Deshpande", date_of_birth="1958-03-12",
        pin_code="411004", city="Pune", mobile="+91 98220 41171",
        email="meera.deshpande@example.in", pan="AFZPD1274K",
        vulnerability_note="Bereavement recorded 14 July 2026 — single point of contact.",
        related={"CUST-4472": "spouse (deceased)"},
    ),
    "CUST-4472": Customer(
        customer_id="CUST-4472", name="Rajesh Deshpande", date_of_birth="1955-11-02",
        pin_code="411004", city="Pune", mobile="+91 98220 41172",
        email="rajesh.deshpande@example.in", pan="AFZPD1275M",
        vulnerability_note="Deceased — claim settlement in progress.",
        related={"CUST-4471": "spouse"},
    ),
    "CUST-8830": Customer(
        customer_id="CUST-8830", name="Suresh Iyer", date_of_birth="1961-06-21",
        pin_code="600028", city="Chennai", mobile="+91 98400 88301",
        email="s.iyer@example.in", pan="BKTPI9932L",
        vulnerability_note="Memory difficulties following a stroke; daughter assists.",
    ),
    "CUST-6612": Customer(
        customer_id="CUST-6612", name="Arjun Malhotra", date_of_birth="1991-01-30",
        pin_code="122009", city="Gurugram", mobile="+91 99100 66121",
        email="arjun.malhotra@example.in", pan="CQWPM4418R",
        vulnerability_note="",
    ),
    "CUST-7788": Customer(
        customer_id="CUST-7788", name="Fatima Sheikh", date_of_birth="1977-09-08",
        pin_code="400051", city="Mumbai", mobile="+91 98200 77881",
        email="f.sheikh@example.in", pan="DLMPS6650N",
        vulnerability_note="Serious illness disclosed 2 August 2026.",
    ),
    "CUST-2205": Customer(
        customer_id="CUST-2205", name="Priya Nair", date_of_birth="1994-04-17",
        pin_code="682024", city="Kochi", mobile="+91 94470 22051",
        email="priya.nair@example.in", pan="EHNPN3308J",
    ),
}

ACCOUNTS: list[Account] = [
    # --- Meera & Rajesh Deshpande: joint home loan, his sole card -----------
    Account(
        account_id="ACC-1001", product=Product.MORTGAGE,
        label="Joint home loan", ifsc="PIBK0004411", account_number="44710021884",
        balance=-2_840_000.00, monthly_payment=24_850.00, next_payment_date="5 September",
        arrears_months=3, arrears_amount=74_550.00,
        holders=["CUST-4471", "CUST-4472"],
        transactions=[
            Transaction(date=_recent(4), description="EMI — auto debit returned unpaid",
                        amount=0.0, category="arrears"),
            Transaction(date=_recent(35), description="EMI — auto debit returned unpaid",
                        amount=0.0, category="arrears"),
        ],
    ),
    Account(
        account_id="ACC-1002", product=Product.SAVINGS,
        label="Joint savings account", ifsc="PIBK0004411", account_number="44710038512",
        balance=18_240.00, holders=["CUST-4471", "CUST-4472"],
        transactions=[
            Transaction(date=_recent(2), description="Pension credit", amount=14_200.00,
                        category="income"),
            Transaction(date=_recent(3), description="MSEDCL electricity", amount=-2_180.00,
                        category="essential"),
            Transaction(date=_recent(6), description="Funeral expenses",
                        amount=-46_000.00, category="one-off"),
        ],
    ),
    Account(
        # Rajesh's sole card. Meera is not a holder — the guardrail case.
        account_id="ACC-1003", product=Product.CREDIT_CARD,
        label="Credit card", ifsc="PIBK0004411", account_number="44720117340",
        balance=-112_400.00, credit_limit=200_000.00, monthly_payment=5_620.00,
        next_payment_date="12 September", holders=["CUST-4472"],
    ),

    # --- Suresh Iyer ---------------------------------------------------------
    Account(
        account_id="ACC-2001", product=Product.SAVINGS,
        label="Savings account", ifsc="PIBK0006002", account_number="88300194627",
        balance=96_480.00, holders=["CUST-8830"],
        transactions=[
            Transaction(date=_recent(1), description="Pension credit", amount=32_400.00,
                        category="income"),
            Transaction(date=_recent(5), description="Metro Water bill", amount=-960.00,
                        category="essential"),
        ],
    ),

    # --- Arjun Malhotra: job loss, home loan + card -------------------------
    Account(
        account_id="ACC-3001", product=Product.MORTGAGE,
        label="Home loan", ifsc="PIBK0001220", account_number="66120043159",
        balance=-4_820_000.00, monthly_payment=41_300.00, next_payment_date="28 August",
        arrears_months=1, arrears_amount=41_300.00, holders=["CUST-6612"],
    ),
    Account(
        account_id="ACC-3002", product=Product.CREDIT_CARD,
        label="Credit card", ifsc="PIBK0001220", account_number="66120118773",
        balance=-186_500.00, credit_limit=200_000.00, monthly_payment=9_325.00,
        next_payment_date="9 September", holders=["CUST-6612"],
        transactions=[
            Transaction(date=_recent(2), description="Big Bazaar groceries", amount=-3_180.00,
                        category="essential"),
            Transaction(date=_recent(9), description="Salary — final settlement",
                        amount=94_000.00, category="income"),
        ],
    ),

    # --- Fatima Sheikh -------------------------------------------------------
    Account(
        account_id="ACC-4001", product=Product.SAVINGS,
        label="Fixed deposit", ifsc="PIBK0004005", account_number="77880250416",
        balance=340_000.00, holders=["CUST-7788"],
    ),
    Account(
        account_id="ACC-4002", product=Product.PERSONAL_LOAN,
        label="Personal loan", ifsc="PIBK0004005", account_number="77880266308",
        balance=-268_000.00, monthly_payment=8_940.00, next_payment_date="3 September",
        holders=["CUST-7788"],
    ),

    # --- Priya Nair ----------------------------------------------------------
    Account(
        account_id="ACC-5001", product=Product.CREDIT_CARD,
        label="Credit card", ifsc="PIBK0006820", account_number="22050311925",
        balance=-24_760.00, credit_limit=150_000.00, monthly_payment=1_240.00,
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
            "ifsc": account.masked_ifsc,
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
            "city": customer.city,
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
             f"{view['customer']['city']}"]
    if view["customer"]["note_on_file"]:
        lines.append(f"Note already on file: {view['customer']['note_on_file']}")
    for account in view["accounts"]:
        parts = [f"{account['label']} {account['reference']}"]
        if account["owed"]:
            parts.append(f"{inr(account['owed'])} outstanding")
        else:
            parts.append(f"{inr(account['balance'])} available")
        if account["monthly_payment"]:
            parts.append(f"{inr(account['monthly_payment'])} due "
                         f"{account['next_payment_date']}")
        if account["arrears_months"]:
            parts.append(f"{account['arrears_months']} instalment(s) overdue "
                         f"({inr(account['arrears_amount'])})")
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
        values.add(account.ifsc)
    for customer in CUSTOMERS.values():
        values.add(customer.mobile.replace(" ", ""))
        values.add(customer.email)
        values.add(customer.date_of_birth)
        values.add(customer.pin_code)
        values.add(customer.pan)
    return {v for v in values if v}
