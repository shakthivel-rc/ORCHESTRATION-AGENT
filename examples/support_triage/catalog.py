"""The flagship ~120-route customer-support catalog (plan §10.3, §9.1).

This module is two things at once, on purpose:

1. **A realistic catalog.** 126 routes across twelve support areas, written the
   way plan §5.7 says to write them — *description-as-prompt*: what the route
   does, when to use it, and, for the confusable ones, when **not** to. Twelve
   ``args_model`` s, ten routes behind an entitlement (``Route.requires``), and
   one pinned ``human_handoff`` that doubles as ``Router(fallback=...)``.
2. **The benchmark fixture.** :data:`GOLD_CASES` is a labelled set of
   ``(query, expected)`` pairs — 61 single-route cases, 5 genuinely ambiguous
   ones labelled :data:`CLARIFY`, and 5 out-of-scope ones labelled
   :data:`ABSTAIN`. ``examples/support_triage/demo.py`` scores a Router against
   it; the [v0.2] eval CLI reads the same list.

At 126 routes the catalog sits above ``shortlist_min_routes=25``, so
``shortlist="auto"`` genuinely retrieves (it does not bypass), and inside the
plan §5.3 band ``25 <= N < 150`` where K=10. That is the whole point of shipping a
catalog this size: the interesting behaviour — retrieval gap vs decision gap,
position-bias shuffling, entitlement filtering, pinned survival — only exists
above the bypass threshold.

**A note on ``metadata["triggers"]``.** Each route carries a tuple of trigger
phrases in :attr:`~switchboard.Route.metadata`, which switchboard treats as
opaque passthrough. They are **not** used for routing by the library; they exist
so ``demo.py`` can build a deterministic, offline stand-in for a model
(``demo.py`` explains exactly what that stub is and is not). A real deployment
carries a handler reference there instead.

Zero extras required: this module imports Pydantic and switchboard, nothing else.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from switchboard import Registry, Route

__all__ = [
    "ABSTAIN",
    "CLARIFY",
    "DOMAINS",
    "ENTITLEMENTS",
    "FALLBACK_ROUTE",
    "GOLD_CASES",
    "AddressChange",
    "CardRef",
    "DateRange",
    "EmailChange",
    "GoldCase",
    "InvoiceRef",
    "OrderItemRef",
    "OrderRef",
    "ProductRef",
    "RefundRequest",
    "SeatChange",
    "SubscriptionChange",
    "TicketRef",
    "registry",
    "routes",
    "triggers_for",
]


# --------------------------------------------------------------------------- #
# Argument models (plan §3.1: args are extracted in the same LLM call).
#
# Required vs optional is load-bearing, not decoration: a missing *required*
# field is what drives plan §3.8's row-4 downgrade from `route` to `clarify` —
# the route choice was sound, the missing argument is a question for the user.
# --------------------------------------------------------------------------- #


class OrderRef(BaseModel):
    """Points at one order."""

    order_id: str


class OrderItemRef(BaseModel):
    """Points at one line item inside an order; the item is often implied."""

    order_id: str
    item_id: str | None = None


class RefundRequest(BaseModel):
    """A refund ask. ``reason`` is optional — most customers volunteer it."""

    order_id: str
    reason: str | None = None


class InvoiceRef(BaseModel):
    """Points at one invoice by its printed number."""

    invoice_id: str


class AddressChange(BaseModel):
    """Re-addresses a shipment that has not left the warehouse."""

    order_id: str
    address: str


class CardRef(BaseModel):
    """Identifies a stored card without ever carrying a full PAN."""

    last4: str | None = None


class DateRange(BaseModel):
    """A reporting window. Both ends optional: "everything" is a valid range."""

    start_date: str | None = None
    end_date: str | None = None


class ProductRef(BaseModel):
    """Points at one catalog product."""

    sku: str


class EmailChange(BaseModel):
    """The new address for an email change."""

    new_email: str


class SeatChange(BaseModel):
    """Seat-count delta for a team subscription."""

    seats: int


class SubscriptionChange(BaseModel):
    """A plan change. Both fields optional — "cancel" needs neither."""

    plan: str | None = None
    effective: str | None = None


class TicketRef(BaseModel):
    """Points at an existing support ticket."""

    ticket_id: str


# --------------------------------------------------------------------------- #
# Route construction helper.
# --------------------------------------------------------------------------- #


def _r(
    name: str,
    description: str,
    examples: tuple[str, ...],
    tags: tuple[str, ...],
    triggers: tuple[str, ...],
    *,
    args: type[BaseModel] | None = None,
    requires: tuple[str, ...] = (),
    pinned: bool = False,
    label: str | None = None,
) -> Route:
    """Build one catalog route. ``tags[0]`` is the route's domain."""
    metadata: dict[str, Any] = {"domain": tags[0], "triggers": list(triggers)}
    return Route(
        name=name,
        description=description,
        args_model=args,
        examples=examples,
        tags=frozenset(tags),
        requires=frozenset(requires),
        pinned=pinned,
        clarify_label=label,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# 1. Billing (14)
# --------------------------------------------------------------------------- #

_BILLING: tuple[Route, ...] = (
    _r(
        "billing_invoice_view",
        "Show one specific invoice the customer names by number. Use when they ask to see, download "
        "or resend a particular bill. Do not use to explain what a charge was for.",
        ("can I see invoice INV-2231", "send me a copy of my bill from March", "download invoice INV-88"),
        ("billing", "invoices"),
        ("see invoice", "view invoice", "copy of my invoice", "invoice pdf", "download invoice"),
        args=InvoiceRef,
    ),
    _r(
        "billing_invoice_list",
        "List the invoices issued over a period. Use for 'show me all my bills' style asks. Do not use "
        "when the customer names a single invoice number.",
        ("list my invoices for last year", "show me my invoice history"),
        ("billing", "invoices"),
        ("list my invoices", "all my invoices", "invoice history", "past invoices"),
        args=DateRange,
    ),
    _r(
        "billing_charge_explain",
        "Explain what an unexpected charge on the account was for. Use when the customer recognises the "
        "payment but not the reason. Do not use when they say they were billed twice.",
        ("why was I charged 49 euros", "what is this charge on my statement"),
        ("billing", "charges"),
        ("why was i charged", "what is this charge", "explain this charge", "unexpected charge",
         "charge on my card"),
        label="explaining a charge",
    ),
    _r(
        "billing_duplicate_charge",
        "Investigate the same amount being taken more than once. Use only when the customer says they "
        "were billed twice or more. Do not use for a single charge they simply do not recognise.",
        ("I was charged twice for the same order", "there are two identical payments"),
        ("billing", "charges"),
        ("charged twice", "double charged", "duplicate charge", "two identical payments"),
    ),
    _r(
        "billing_update_payment_method",
        "Replace the card or bank details held on file. Use when the customer wants to pay with "
        "something new. Do not use when an existing payment has just failed.",
        ("I need to update my card on file", "change my payment method to a new card"),
        ("billing", "payment_methods"),
        ("update my card", "change payment method", "new credit card", "card on file"),
        args=CardRef,
    ),
    _r(
        "billing_remove_payment_method",
        "Delete a stored card or bank mandate. Use when the customer wants a payment method gone. Do "
        "not use to swap one card for another — that is an update.",
        ("remove my old card", "delete the payment method ending 4242"),
        ("billing", "payment_methods"),
        ("remove my card", "delete payment method", "remove the card ending"),
        args=CardRef,
    ),
    _r(
        "billing_change_billing_address",
        "Change the address that appears on invoices and is used for tax. Use for the address on the "
        "bill. Do not use for where a parcel should be delivered.",
        ("change my billing address", "my invoices show the old address"),
        ("billing", "addresses"),
        ("billing address", "change my billing address", "address on my invoice", "change my address"),
        label="your billing address",
    ),
    _r(
        "billing_change_billing_cycle",
        "Move the account between monthly and annual billing. Use when the customer wants to be "
        "invoiced on a different rhythm. Do not use to change which plan they are on.",
        ("bill me monthly instead", "switch to annual billing"),
        ("billing", "cycles"),
        ("billing cycle", "bill me monthly", "switch to annual billing", "billed annually"),
    ),
    _r(
        "billing_payment_failed",
        "Diagnose and retry a payment the bank refused. Use when a charge was declined or a renewal "
        "failed. Do not use when the payment succeeded but looks wrong.",
        ("my payment failed", "the card was declined at renewal"),
        ("billing", "charges"),
        ("payment failed", "payment declined", "card declined", "payment did not go through"),
    ),
    _r(
        "billing_request_receipt",
        "Issue a receipt or proof of payment for a completed order. Use when the customer needs "
        "evidence they paid. Do not use when they want the full VAT invoice document.",
        ("can I get a receipt for order 8812", "I need proof of payment"),
        ("billing", "invoices"),
        ("receipt for order", "proof of payment", "send me a receipt"),
        args=OrderRef,
    ),
    _r(
        "billing_tax_exemption",
        "Apply a tax or VAT exemption to a business account. Use when the customer says they should "
        "not be taxed. Requires a business account.",
        ("we are tax exempt", "apply our VAT exemption certificate"),
        ("billing", "tax"),
        ("tax exempt", "vat exemption", "tax exemption certificate"),
        requires=("business_account",),
    ),
    _r(
        "billing_vat_id_update",
        "Record or correct the VAT/tax identification number printed on invoices. Use for the number "
        "itself. Do not use to claim an exemption.",
        ("add our VAT number to invoices", "my tax id is wrong on the bill"),
        ("billing", "tax"),
        ("vat number", "vat id", "tax id on my invoice"),
    ),
    _r(
        "billing_export_ledger",
        "Export the full billing ledger as a file for accounting. Use for bookkeeping exports over a "
        "period. Requires a billing administrator.",
        ("export the billing ledger for last quarter", "I need the accounting export as CSV"),
        ("billing", "reporting"),
        ("export the billing ledger", "accounting export", "billing csv export"),
        args=DateRange,
        requires=("billing_admin",),
    ),
    _r(
        "billing_dispute_charge",
        "Open a formal dispute or chargeback on a payment the customer rejects. Use when they contest "
        "the charge rather than merely asking about it. Do not use for a simple explanation request.",
        ("I want to dispute this charge", "I am going to file a chargeback"),
        ("billing", "charges"),
        ("dispute a charge", "dispute this charge", "chargeback", "contest a charge",
         "charge on my card"),
        label="disputing a charge",
    ),
)


# --------------------------------------------------------------------------- #
# 2. Payments (10)
# --------------------------------------------------------------------------- #

_PAYMENTS: tuple[Route, ...] = (
    _r(
        "payments_installment_plan",
        "Set up paying for an order in instalments. Use when the customer asks to spread a cost. Do "
        "not use for changing the subscription billing cycle.",
        ("can I pay in installments", "is there a payment plan for this"),
        ("payments", "plans"),
        ("pay in installments", "payment plan", "split the payment", "pay in instalments"),
    ),
    _r(
        "payments_apply_credit",
        "Apply existing store credit to an order or the next invoice. Use when the customer has a "
        "credit balance to spend. Do not use for gift cards, which have their own code.",
        ("apply my store credit to this order", "use my credit balance"),
        ("payments", "credits"),
        ("apply my store credit", "use my credit balance", "store credit"),
    ),
    _r(
        "payments_gift_card_redeem",
        "Redeem a gift card code onto the account. Use when the customer has a code to enter. Do not "
        "use when they only want to know the remaining balance.",
        ("redeem a gift card", "here is my gift card code"),
        ("payments", "gift_cards"),
        ("redeem a gift card", "gift card code", "add a gift card"),
    ),
    _r(
        "payments_gift_card_balance",
        "Report the remaining value on a gift card. Use for balance questions only. Do not use to "
        "apply the card to a purchase.",
        ("what is my gift card balance", "how much is left on my gift card"),
        ("payments", "gift_cards"),
        ("gift card balance", "how much is left on my gift card"),
    ),
    _r(
        "payments_promo_code_apply",
        "Apply a working promotional or discount code to a cart or order. Use when the code is valid "
        "and just needs applying. Do not use when the customer reports the code failing.",
        ("apply promo code SPRING20", "I have a discount code to use"),
        ("payments", "promotions"),
        ("promo code", "discount code", "coupon code", "apply promo"),
    ),
    _r(
        "payments_promo_code_invalid",
        "Investigate a promotional code that is refused at checkout. Use when the customer says the "
        "code does not work. Do not use when they simply want to apply a valid code.",
        ("my promo code is not working", "the coupon was rejected at checkout"),
        ("payments", "promotions"),
        ("promo code is not working", "coupon was rejected", "discount code invalid",
         "code is not working"),
    ),
    _r(
        "payments_currency_change",
        "Change the currency the account is charged in. Use for currency questions. Do not use for "
        "tax or VAT questions.",
        ("can I pay in euros instead", "change my currency to GBP"),
        ("payments", "currency"),
        ("change currency", "pay in euros", "different currency", "change my currency"),
    ),
    _r(
        "payments_wire_transfer_setup",
        "Arrange payment by bank transfer instead of card. Use when the customer wants to be invoiced "
        "and pay by wire. Do not use for instalments on a card.",
        ("can we pay by wire transfer", "please invoice us for bank transfer"),
        ("payments", "methods"),
        ("pay by wire", "bank transfer", "wire transfer"),
    ),
    _r(
        "payments_authorization_hold",
        "Explain a pending authorisation hold that is not yet a real charge. Use when the customer "
        "sees a pending amount. Do not use when the money has actually left the account.",
        ("there is a pending authorization on my card", "why is there a hold on my card"),
        ("payments", "charges"),
        ("pending authorization", "hold on my card", "pending charge", "authorisation hold"),
    ),
    _r(
        "payments_refund_status",
        "Report where an already-approved refund is in the banking pipeline. Use when a refund was "
        "promised but has not landed. Do not use to request a new refund.",
        ("where is my refund", "the refund has not arrived yet"),
        ("payments", "refunds"),
        ("where is my refund", "refund status", "refund has not arrived", "about a refund"),
        label="chasing a refund you were already promised",
    ),
)


# --------------------------------------------------------------------------- #
# 3. Shipping (14)
# --------------------------------------------------------------------------- #

_SHIPPING: tuple[Route, ...] = (
    _r(
        "shipping_track_parcel",
        "Give the current carrier tracking status of a shipped order. Use for 'where is my package'. "
        "Do not use when the parcel is confirmed lost or the delivery date is the real question.",
        ("where is my package for order 4471", "do you have a tracking number", "has my order shipped"),
        ("shipping", "tracking"),
        ("where is my package", "track my order", "tracking number", "has my order shipped"),
        args=OrderRef,
    ),
    _r(
        "shipping_delivery_estimate",
        "Give the expected delivery date for an order that is on time. Use for 'when will it arrive'. "
        "Do not use once the promised date has already passed.",
        ("when will it arrive, order 8823", "what is the expected delivery date"),
        ("shipping", "tracking"),
        ("when will it arrive", "delivery estimate", "expected delivery date", "when does it arrive"),
        args=OrderRef,
    ),
    _r(
        "shipping_change_address",
        "Re-address a parcel that has not yet been dispatched. Use for where the goods should go. Do "
        "not use for the address printed on the invoice.",
        ("change the delivery address for order 3301", "ship it somewhere else"),
        ("shipping", "addresses"),
        ("change the delivery address", "ship it somewhere else", "wrong shipping address",
         "change my address"),
        args=AddressChange,
        label="your delivery address",
    ),
    _r(
        "shipping_delayed_parcel",
        "Handle a parcel that is late but still in the carrier network. Use once the promised date has "
        "passed. Do not use when the carrier has marked the parcel as delivered.",
        ("my parcel is late for order 3390", "this shipment is taking far too long"),
        ("shipping", "problems"),
        ("my parcel is late", "delayed shipment", "package is taking too long", "taking far too long"),
        args=OrderRef,
    ),
    _r(
        "shipping_lost_parcel",
        "Open a lost-in-transit claim. Use when the parcel never arrived or tracking has gone silent "
        "for days. Do not use when it arrived but was damaged.",
        ("the package never arrived for order 5521", "my parcel is lost"),
        ("shipping", "problems"),
        ("package never arrived", "lost parcel", "parcel is missing", "never arrived"),
        args=OrderRef,
    ),
    _r(
        "shipping_damaged_parcel",
        "Handle goods that arrived damaged in transit. Use when the packaging or contents were harmed "
        "on the way. Do not use for a product that was faulty out of the box.",
        ("order 6640 arrived damaged, the box was crushed", "the parcel was smashed"),
        ("shipping", "problems"),
        ("arrived damaged", "box was crushed", "damaged in transit", "parcel was smashed"),
        args=OrderRef,
    ),
    _r(
        "shipping_missing_item",
        "Handle a delivery that arrived short of what was ordered. Use when part of the order is "
        "absent. Do not use when the whole parcel is missing.",
        ("one item missing from my order 3312", "my delivery was incomplete"),
        ("shipping", "problems"),
        ("item missing from my order", "one item did not arrive", "incomplete delivery",
         "delivery was incomplete"),
        args=OrderItemRef,
    ),
    _r(
        "shipping_upgrade_speed",
        "Upgrade an unshipped order to a faster service. Use when the customer wants it sooner. Do not "
        "use for a parcel that is already in transit.",
        ("can I upgrade to express for order 5540", "please expedite my delivery"),
        ("shipping", "options"),
        ("faster shipping", "upgrade to express", "expedite my delivery", "next day delivery"),
        args=OrderRef,
    ),
    _r(
        "shipping_redeliver",
        "Book a second delivery attempt after a missed one. Use when the courier could not hand the "
        "parcel over. Do not use to redirect to a different address.",
        ("schedule a redelivery for order 2233", "I missed the delivery yesterday"),
        ("shipping", "options"),
        ("schedule a redelivery", "redelivery", "missed the delivery"),
        args=OrderRef,
    ),
    _r(
        "shipping_pickup_point",
        "Send a parcel to a locker or parcel shop instead of a home address. Use for collection-point "
        "requests. Do not use to change the street address.",
        ("can it go to a pickup point", "deliver to a parcel locker"),
        ("shipping", "options"),
        ("pickup point", "parcel locker", "parcel shop", "collect from a locker"),
    ),
    _r(
        "shipping_customs_fees",
        "Explain import duties, customs charges and parcels held at the border. Use for cross-border "
        "fees. Do not use for domestic delivery costs.",
        ("held at customs, what is the import duty", "why do I owe a customs fee"),
        ("shipping", "international"),
        ("customs fee", "import duty", "held at customs", "customs charge"),
    ),
    _r(
        "shipping_international_availability",
        "Say whether an address or country can be shipped to at all. Use for 'do you deliver to X'. Do "
        "not use for how much shipping to X costs.",
        ("do you ship to Norway", "is international shipping available for my country"),
        ("shipping", "international"),
        ("do you ship to", "international shipping", "ship abroad", "do you deliver to"),
    ),
    _r(
        "shipping_carrier_change",
        "Move an unshipped order to a different courier. Use when the customer objects to the assigned "
        "carrier. Do not use to change service speed.",
        ("can you use a different carrier for order 7120", "please send it with another courier"),
        ("shipping", "options"),
        ("different carrier", "another courier", "change the carrier"),
        args=OrderRef,
    ),
    _r(
        "shipping_cost_explain",
        "Explain how a delivery charge was calculated. Use for questions about the price of shipping. "
        "Do not use for customs duties.",
        ("why is delivery so expensive", "how is the shipping cost worked out"),
        ("shipping", "pricing"),
        ("shipping cost", "why is delivery so expensive", "postage price", "delivery charge"),
    ),
)


# --------------------------------------------------------------------------- #
# 4. Returns & refunds (13)
# --------------------------------------------------------------------------- #

_RETURNS: tuple[Route, ...] = (
    _r(
        "returns_start_return",
        "Open a return for goods the customer wants to send back. Use to begin the process. Do not use "
        "when they want a different item instead — that is an exchange.",
        ("I want to start a return for order 7781", "how do I send this back"),
        ("returns", "process"),
        ("start a return", "i want to return", "send this back", "open a return"),
        args=OrderItemRef,
    ),
    _r(
        "returns_print_label",
        "Issue the prepaid postage label for an approved return. Use when the return exists and the "
        "label is what is missing. Do not use to open a new return.",
        ("I need a return label for order 1180", "resend the prepaid postage label"),
        ("returns", "process"),
        ("return label", "postage label", "label for my return"),
        args=OrderRef,
    ),
    _r(
        "returns_policy_explain",
        "Explain the returns window, condition rules and exclusions. Use for policy questions in the "
        "abstract. Do not use when the customer is returning a specific order.",
        ("what is your return policy", "how long do I have to return something"),
        ("returns", "policy"),
        ("return policy", "how long do i have to return", "returns window"),
    ),
    _r(
        "returns_status",
        "Report where a return is in processing after it was posted. Use for 'have you got it yet'. Do "
        "not use for where the refund money is.",
        ("have you received my return for order 2004", "what is the status of my return"),
        ("returns", "process"),
        ("received my return", "return status", "status of my return"),
        args=OrderRef,
    ),
    _r(
        "returns_exchange_item",
        "Swap a delivered item for a different size, colour or variant. Use when the customer wants "
        "replacement goods rather than money. Do not use for a plain refund.",
        ("can I exchange it for another size, order 8890", "swap this for the blue one"),
        ("returns", "exchanges"),
        ("exchange it", "swap for another size", "different colour instead", "swap this for"),
    ),
    _r(
        "returns_refund_request",
        "Request money back for an order. Use when the customer wants a refund they have not yet been "
        "promised. Do not use to chase a refund already approved.",
        ("I want my money back for order 123", "please refund me", "can I get a refund"),
        ("returns", "refunds"),
        ("i want my money back", "request a refund", "refund me", "get a refund", "about a refund"),
        args=RefundRequest,
        label="requesting a new refund",
    ),
    _r(
        "returns_refund_amount_wrong",
        "Investigate a refund that arrived for the wrong amount. Use when money came back but it is "
        "short or over. Do not use when nothing has arrived at all.",
        ("my refund is short by 12 euros", "you refunded the wrong amount for order 4402"),
        ("returns", "refunds"),
        ("refund is short", "refunded the wrong amount", "partial refund", "wrong refund amount"),
        args=RefundRequest,
    ),
    _r(
        "returns_warranty_claim",
        "Open a manufacturer warranty claim for a product still in its warranty period. Use for repair "
        "or replacement under warranty. Do not use inside the normal returns window.",
        ("warranty claim for sku AB-99", "is this still under warranty"),
        ("returns", "warranty"),
        ("warranty claim", "under warranty", "warranty repair"),
        args=ProductRef,
    ),
    _r(
        "returns_faulty_item",
        "Handle goods that do not work as sold. Use when the product itself is defective. Do not use "
        "when the damage happened in transit.",
        ("the product is defective for order 9001, it stopped working", "this item is faulty"),
        ("returns", "problems"),
        ("stopped working", "faulty item", "is defective", "item is faulty", "broken on arrival"),
        args=OrderItemRef,
    ),
    _r(
        "returns_wrong_item",
        "Handle the wrong product being shipped. Use when what arrived is not what was ordered. Do not "
        "use when the right product arrived damaged.",
        ("this is not what I ordered, wrong item in order 7712", "you sent the wrong product"),
        ("returns", "problems"),
        ("wrong item", "not what i ordered", "sent the wrong product", "order is wrong"),
        args=OrderItemRef,
        label="receiving the wrong item",
    ),
    _r(
        "returns_late_return",
        "Consider a return requested after the window closed. Use when the deadline has passed. Do not "
        "use inside the normal window.",
        ("I am past the return window, can you help", "I missed the return deadline"),
        ("returns", "policy"),
        ("past the return window", "return it late", "missed the return deadline"),
    ),
    _r(
        "returns_pickup_arrange",
        "Book a courier collection of returned goods from the customer's address. Use for bulky items "
        "or when posting is impractical. Do not use to issue a postage label.",
        ("can you collect the return from my address, order 5510", "arrange a pickup for my return"),
        ("returns", "process"),
        ("collect the return", "arrange a pickup for my return", "pick up the return"),
        args=OrderRef,
    ),
    _r(
        "returns_restocking_fee",
        "Explain or contest a restocking fee deducted from a refund. Use when a fee was charged on a "
        "return. Do not use for a refund that is simply late.",
        ("why was I charged a restocking fee", "there is a return fee on my refund"),
        ("returns", "fees"),
        ("restocking fee", "return fee"),
    ),
)


# --------------------------------------------------------------------------- #
# 5. Orders (12)
# --------------------------------------------------------------------------- #

_ORDERS: tuple[Route, ...] = (
    _r(
        "orders_status",
        "Report the fulfilment state of an order before it ships. Use for 'what is happening with my "
        "order'. Do not use once it is with the carrier — that is tracking.",
        ("what is the status of order 9912", "is my order being packed yet"),
        ("orders", "status"),
        ("order status", "status of order", "status of my order", "happening with my order"),
        args=OrderRef,
    ),
    _r(
        "orders_cancel",
        "Cancel an order that has not shipped. Use for cancelling a purchase. Do not use to cancel a "
        "recurring subscription.",
        ("cancel my order 2210", "I changed my mind, stop the order"),
        ("orders", "changes"),
        ("cancel my order", "stop the order", "cancel the order", "cancel it"),
        args=OrderRef,
        label="cancelling an order",
    ),
    _r(
        "orders_modify_items",
        "Add, remove or change quantities on an order that has not shipped. Use for edits to what was "
        "bought. Do not use to cancel the whole order.",
        ("add an item to my order 4471", "remove one item from order 4471"),
        ("orders", "changes"),
        ("add an item to my order", "change my order", "remove an item from my order",
         "order is wrong"),
        args=OrderItemRef,
        label="changing what is in an order",
    ),
    _r(
        "orders_reorder",
        "Recreate a previous order as a new one. Use when the customer wants the same goods again. Do "
        "not use to modify the original order.",
        ("I want to reorder the same thing", "order 3320 again please"),
        ("orders", "purchase"),
        ("reorder", "order it again", "buy the same thing again", "same thing again"),
        args=OrderRef,
    ),
    _r(
        "orders_history",
        "List the customer's past orders. Use for 'what have I bought'. Do not use when they name one "
        "specific order.",
        ("show me my order history", "what have I bought this year"),
        ("orders", "status"),
        ("order history", "my past orders", "what have i bought"),
        args=DateRange,
    ),
    _r(
        "orders_place_order",
        "Place a brand new order for a named product. Use when the customer is buying now. Do not use "
        "for questions about stock or price alone.",
        ("place an order for sku QR-14", "I want to buy two of these"),
        ("orders", "purchase"),
        ("place an order", "i want to buy", "order this for me"),
        args=ProductRef,
    ),
    _r(
        "orders_split_shipment",
        "Explain why one order arrived in several parcels, or request that it does. Use for split "
        "deliveries. Do not use when an item is genuinely missing.",
        ("why did my order arrive in parts", "can you split the shipment for order 8801"),
        ("orders", "fulfilment"),
        ("split shipment", "two separate deliveries", "arrive in parts", "arrived in parts"),
        args=OrderRef,
    ),
    _r(
        "orders_backorder_status",
        "Report when a backordered product will be allocated. Use when the item is out of stock but "
        "already paid for. Do not use for stock questions before buying.",
        ("my item is on backorder, when will it ship", "when does sku LM-12 come back for my order"),
        ("orders", "fulfilment"),
        ("on backorder", "backordered", "back order status"),
        args=ProductRef,
    ),
    _r(
        "orders_preorder_manage",
        "Manage a pre-order: release date, changes or cancellation. Use for goods not yet released. Do "
        "not use for in-stock items.",
        ("what is the release date for my preorder", "cancel my pre-order of sku ZX-1"),
        ("orders", "fulfilment"),
        ("preorder", "pre order", "release date for my"),
        args=ProductRef,
    ),
    _r(
        "orders_bulk_quote",
        "Produce a priced quotation for a large or wholesale order. Use for volume purchases. Requires "
        "a business account.",
        ("I need a bulk quote for 500 units", "can we get wholesale pricing"),
        ("orders", "b2b"),
        ("bulk quote", "wholesale pricing", "quote for", "volume pricing"),
        requires=("business_account",),
    ),
    _r(
        "orders_gift_wrap",
        "Add gift wrapping or a gift message to an order. Use for presentation requests. Do not use to "
        "change the delivery address.",
        ("add gift wrap and a gift message to order 4410", "can you wrap it as a present"),
        ("orders", "options"),
        ("gift wrap", "gift message", "wrap it as a present"),
        args=OrderRef,
    ),
    _r(
        "orders_invoice_for_order",
        "Produce the VAT invoice document for one order. Use when the customer needs the formal "
        "invoice for a purchase. Do not use to list all invoices.",
        ("send me the invoice for order 5502", "I need an order invoice for accounting"),
        ("orders", "invoices"),
        ("invoice for order", "order invoice", "invoice for this order"),
        args=OrderRef,
    ),
)


# --------------------------------------------------------------------------- #
# 6. Account (14)
# --------------------------------------------------------------------------- #

_ACCOUNT: tuple[Route, ...] = (
    _r(
        "account_login_problem",
        "Diagnose a customer who cannot get into their account. Use for sign-in failures. Do not use "
        "when they know the password is wrong and simply want a reset.",
        ("I cannot log in to my account", "sign in is not working for me"),
        ("account", "access"),
        ("cannot log in", "can not log in", "cannot sign in", "sign in is not working",
         "login not working"),
    ),
    _r(
        "account_password_reset",
        "Send a password reset or change the password. Use when the credential itself is the problem. "
        "Do not use for two-factor lockouts.",
        ("reset my password please", "I forgot my password"),
        ("account", "access"),
        ("reset my password", "forgot my password", "change my password", "password reset"),
    ),
    _r(
        "account_change_email",
        "Change the email address the account signs in with and receives mail at. Use for the address "
        "itself. Do not use to stop marketing mail.",
        ("change my email to sam@example.com", "update my email address"),
        ("account", "profile"),
        ("change my email", "update my email address", "new email address"),
        args=EmailChange,
    ),
    _r(
        "account_change_phone",
        "Change the phone number on the account. Use for the number itself. Do not use for two-factor "
        "recovery.",
        ("change my phone number", "update my mobile number"),
        ("account", "profile"),
        ("change my phone number", "update my mobile number", "new phone number"),
    ),
    _r(
        "account_two_factor_setup",
        "Turn on two-factor authentication or change the second factor. Use when setting it up. Do not "
        "use when the customer is locked out.",
        ("set up two factor authentication", "I want to use an authenticator app"),
        ("account", "security"),
        ("two factor", "2fa setup", "authenticator app", "set up two factor"),
    ),
    _r(
        "account_two_factor_recovery",
        "Recover an account locked out by a lost second factor. Use when the customer cannot complete "
        "2FA. Do not use for a simple password reset.",
        ("I lost my authenticator, 2fa recovery", "locked out of two factor"),
        ("account", "security"),
        ("lost my authenticator", "2fa recovery", "locked out of two factor", "lost my 2fa"),
    ),
    _r(
        "account_close",
        "Close the account permanently. Use when the customer wants the account itself gone. Requires "
        "the account owner. Do not use to cancel a subscription only.",
        ("close my account", "I want to delete my account"),
        ("account", "lifecycle"),
        ("close my account", "delete my account", "shut down my account"),
        requires=("account_owner",),
    ),
    _r(
        "account_transfer_ownership",
        "Move ownership of the account to another user. Use for handovers. Requires the account owner.",
        ("transfer ownership to my colleague", "hand the account to someone else"),
        ("account", "lifecycle"),
        ("transfer ownership", "hand the account to someone else", "change the account owner"),
        requires=("account_owner",),
    ),
    _r(
        "account_add_user",
        "Invite an additional user onto the account. Use to grant someone access. Do not use to buy "
        "more subscription seats.",
        ("invite a colleague to the workspace", "add a team member to our account"),
        ("account", "team"),
        ("invite a colleague", "add a user", "add a team member", "invite someone"),
    ),
    _r(
        "account_remove_user",
        "Remove a user's access from the account. Use to revoke access. Do not use to reduce the "
        "number of paid seats.",
        ("remove a user from our account", "revoke access for dana@example.com"),
        ("account", "team"),
        ("remove a user", "revoke access for", "delete a team member"),
    ),
    _r(
        "account_permissions_change",
        "Change what an existing user is allowed to do. Use for roles and permissions. Do not use to "
        "add or remove the user entirely.",
        ("make dana an admin", "change the permissions for this user"),
        ("account", "team"),
        ("change permissions", "make them an admin", "make dana an admin", "change the role for"),
    ),
    _r(
        "account_notification_preferences",
        "Change which transactional notifications the account receives. Use for notification settings. "
        "Do not use for marketing opt-out, which is a privacy request.",
        ("stop emailing me about every order", "change my notification settings"),
        ("account", "preferences"),
        ("notification settings", "stop emailing me", "email preferences", "notification preferences"),
    ),
    _r(
        "account_language_preference",
        "Change the interface or correspondence language. Use for language settings. Do not use for "
        "currency.",
        ("change the language to French", "I want the interface in German"),
        ("account", "preferences"),
        ("change the language", "interface language", "in another language"),
    ),
    _r(
        "account_merge_duplicate",
        "Merge two accounts belonging to the same customer. Use when duplicates exist. Do not use to "
        "transfer ownership between different people.",
        ("I have two accounts, please merge them", "there is a duplicate account for my email"),
        ("account", "lifecycle"),
        ("merge my accounts", "two accounts", "duplicate account"),
    ),
)


# --------------------------------------------------------------------------- #
# 7. Technical (14)
# --------------------------------------------------------------------------- #

_TECHNICAL: tuple[Route, ...] = (
    _r(
        "tech_app_crash",
        "Investigate the mobile or desktop app closing unexpectedly. Use for crashes. Do not use for "
        "errors shown on the website.",
        ("the app crashes on startup", "the app keeps closing when I open orders"),
        ("technical", "apps"),
        ("app crashes", "app keeps closing", "crashes on startup", "the app crashed"),
    ),
    _r(
        "tech_page_error",
        "Investigate an error page or error message on the website. Use when the site shows an error. "
        "Do not use when the site merely feels slow.",
        ("I get a 500 error on the account page", "something went wrong message keeps showing"),
        ("technical", "web"),
        ("500 error", "error message on the page", "something went wrong page", "error page"),
    ),
    _r(
        "tech_slow_performance",
        "Investigate slowness without an outright failure. Use when pages load but too slowly. Do not "
        "use when an error is shown.",
        ("the site is very slow today", "pages are loading forever"),
        ("technical", "web"),
        ("very slow", "site is slow", "loading forever", "extremely slow"),
    ),
    _r(
        "tech_checkout_broken",
        "Investigate a checkout that cannot be completed for technical reasons. Use when the customer "
        "cannot pay at all. Do not use when the card was declined by the bank.",
        ("checkout does not work", "the pay button does nothing"),
        ("technical", "web"),
        ("checkout does not work", "cannot complete checkout", "checkout button", "pay button does"),
    ),
    _r(
        "tech_search_broken",
        "Investigate site search returning nothing or wrong results. Use for search failures. Do not "
        "use for stock availability questions.",
        ("search returns nothing for anything I type", "the search is broken"),
        ("technical", "web"),
        ("search returns nothing", "search is broken", "search does not work"),
    ),
    _r(
        "tech_email_not_received",
        "Investigate a transactional email that never arrived. Use for missing confirmations, resets "
        "or receipts. Do not use for marketing preferences.",
        ("I did not receive the email confirmation", "no confirmation email came through"),
        ("technical", "email"),
        ("did not receive the email", "no confirmation email", "confirmation email missing",
         "no email came"),
    ),
    _r(
        "tech_file_upload_failure",
        "Investigate uploads that fail or hang. Use for attachment and document upload problems. Do "
        "not use for downloads or exports.",
        ("my file upload fails every time", "I cannot upload a photo to the claim"),
        ("technical", "web"),
        ("upload fails", "cannot upload", "file upload error", "upload keeps failing"),
    ),
    _r(
        "tech_api_key_rotate",
        "Issue a new API key and revoke the old one. Use when a key must be replaced or is exposed. "
        "Requires developer access.",
        ("rotate my api key, it was compromised", "issue a new API key for production"),
        ("technical", "api"),
        ("rotate my api key", "new api key", "api key compromised", "revoke my api key"),
        requires=("developer",),
    ),
    _r(
        "tech_api_rate_limit",
        "Explain or raise API rate limits. Use for 429s and throttling. Requires developer access. Do "
        "not use for authentication errors.",
        ("we are getting rate limited with 429s", "can you raise our API rate limit"),
        ("technical", "api"),
        ("rate limited", "rate limit", "too many requests", "429"),
        requires=("developer",),
    ),
    _r(
        "tech_webhook_configure",
        "Configure or debug webhook delivery to a callback URL. Use for webhook problems. Requires "
        "developer access.",
        ("my webhook is not firing", "update the callback url for our webhooks"),
        ("technical", "api"),
        ("webhook", "callback url", "webhook not firing", "webhook is not firing"),
        requires=("developer",),
    ),
    _r(
        "tech_integration_setup",
        "Help connect a third-party integration such as a storefront or automation tool. Use for "
        "connecting external systems. Do not use for raw API keys.",
        ("help me connect my Shopify store", "how do I set up the Zapier integration"),
        ("technical", "integrations"),
        ("connect my shopify", "integration setup", "connect to zapier", "set up the integration"),
    ),
    _r(
        "tech_export_data_file",
        "Export operational data as a downloadable file. Use for CSV and report downloads. Do not use "
        "for a GDPR personal-data request.",
        ("export my order data as a csv", "I want to download a data export"),
        ("technical", "exports"),
        ("export my data", "download a csv", "data export", "export as a csv"),
        args=DateRange,
    ),
    _r(
        "tech_browser_compatibility",
        "Investigate behaviour that differs between browsers or devices. Use when it works in one "
        "browser but not another. Do not use for a site-wide outage.",
        ("it does not work in Safari but is fine in Chrome", "the page is broken on my old browser"),
        ("technical", "web"),
        ("does not work in safari", "browser compatibility", "works in chrome but not", "in safari"),
    ),
    _r(
        "tech_report_bug",
        "Record a reproducible defect report for engineering. Use when the customer is reporting a bug "
        "rather than asking for help. Do not use for account or billing problems.",
        ("I want to report a bug in the order form", "I think I found a bug"),
        ("technical", "quality"),
        ("report a bug", "found a bug", "there is a bug"),
        args=TicketRef,
    ),
)


# --------------------------------------------------------------------------- #
# 8. Subscriptions (12)
# --------------------------------------------------------------------------- #

_SUBSCRIPTIONS: tuple[Route, ...] = (
    _r(
        "subs_upgrade_plan",
        "Move the subscription to a higher tier. Use when the customer wants more. Do not use to add "
        "seats on the same tier.",
        ("I want to upgrade my plan to pro", "move us to the higher tier"),
        ("subscriptions", "plans"),
        ("upgrade my plan", "move to the pro plan", "higher tier", "upgrade to pro"),
        args=SubscriptionChange,
    ),
    _r(
        "subs_downgrade_plan",
        "Move the subscription to a lower tier. Use when the customer wants to spend less but stay. Do "
        "not use when they want to leave entirely.",
        ("downgrade my plan to basic", "we need a cheaper plan"),
        ("subscriptions", "plans"),
        ("downgrade my plan", "cheaper plan", "move to the basic plan", "downgrade to basic"),
        args=SubscriptionChange,
    ),
    _r(
        "subs_cancel",
        "Cancel the recurring subscription. Use to end the membership. Do not use to cancel a one-off "
        "product order.",
        ("cancel my subscription", "I want to end my membership"),
        ("subscriptions", "lifecycle"),
        ("cancel my subscription", "end my membership", "stop my plan", "cancel my membership",
         "cancel it"),
        args=SubscriptionChange,
        label="cancelling your subscription",
    ),
    _r(
        "subs_pause",
        "Pause the subscription for a stated period, keeping the account. Use for temporary breaks. Do "
        "not use for permanent cancellation.",
        ("pause my subscription for two months", "can I freeze my membership over summer"),
        ("subscriptions", "lifecycle"),
        ("pause my subscription", "put my plan on hold", "freeze my membership", "pause my plan"),
        args=SubscriptionChange,
    ),
    _r(
        "subs_resume",
        "Restart a paused or cancelled subscription. Use to bring service back. Do not use to change "
        "tier at the same time.",
        ("resume my subscription now", "please reactivate my plan"),
        ("subscriptions", "lifecycle"),
        ("resume my subscription", "reactivate my plan", "unpause", "restart my subscription"),
        args=SubscriptionChange,
    ),
    _r(
        "subs_renewal_date",
        "Say when the subscription next renews and for how much. Use for date and amount questions. Do "
        "not use to change auto-renewal.",
        ("when does it renew", "what is my next billing date"),
        ("subscriptions", "billing"),
        ("when does it renew", "renewal date", "next billing date", "when do i get billed"),
    ),
    _r(
        "subs_auto_renew_toggle",
        "Turn automatic renewal on or off while keeping the current term. Use for the renewal switch "
        "itself. Do not use to cancel immediately.",
        ("turn off auto renew", "please enable automatic renewal again"),
        ("subscriptions", "billing"),
        ("auto renew", "automatic renewal", "turn off auto renew", "stop automatic renewal"),
    ),
    _r(
        "subs_seats_change",
        "Add or remove paid seats on a team subscription. Use for licence counts. Do not use to invite "
        "a user who already has a seat.",
        ("add 3 more seats", "we need to reduce our seats to five"),
        ("subscriptions", "seats"),
        ("add seats", "more seats", "more licences", "reduce seats", "extra seats"),
        args=SeatChange,
    ),
    _r(
        "subs_trial_extend",
        "Extend a free trial that is ending or has just ended. Use for more trial time. Do not use to "
        "convert to a paid plan.",
        ("can you extend my trial by a week", "my trial ended too soon"),
        ("subscriptions", "trials"),
        ("extend my trial", "longer trial", "trial ended too soon", "more trial time"),
    ),
    _r(
        "subs_trial_convert",
        "Convert a trial into a paid subscription. Use when the customer is ready to pay. Do not use "
        "to extend the trial.",
        ("convert my trial to a paid plan", "I want to start paying after the trial"),
        ("subscriptions", "trials"),
        ("convert my trial", "start paying after the trial", "turn my trial into"),
        args=SubscriptionChange,
    ),
    _r(
        "subs_plan_compare",
        "Explain the differences between plans so the customer can choose. Use for pre-purchase "
        "comparisons. Do not use when they have already decided to move.",
        ("what is the difference between the plans", "which plan should I pick"),
        ("subscriptions", "plans"),
        ("difference between the plans", "which plan should i", "compare plans", "compare the plans"),
    ),
    _r(
        "subs_student_discount",
        "Apply an educational or student discount to a subscription. Use for eligibility and "
        "application. Do not use for promotional codes.",
        ("do you offer a student discount", "we need academic pricing for our school"),
        ("subscriptions", "pricing"),
        ("student discount", "academic pricing", "educational discount"),
    ),
)


# --------------------------------------------------------------------------- #
# 9. Product catalog (8)
# --------------------------------------------------------------------------- #

_PRODUCT: tuple[Route, ...] = (
    _r(
        "product_availability",
        "Say whether a product is currently in stock. Use before purchase. Do not use for an item "
        "already ordered and backordered.",
        ("is sku QR-14 in stock", "do you have this in medium"),
        ("product", "stock"),
        ("in stock", "do you have this in", "still available", "availability of"),
        args=ProductRef,
    ),
    _r(
        "product_restock_alert",
        "Register the customer for a notification when an out-of-stock product returns. Use when they "
        "want to be told later. Do not use to answer a stock question now.",
        ("notify me when sku QR-14 is back in stock", "set a restock alert for this"),
        ("product", "stock"),
        ("notify me when", "restock alert", "tell me when it is available", "let me know when it is back"),
        args=ProductRef,
    ),
    _r(
        "product_specification",
        "Give the technical specification of a product: dimensions, materials, power. Use for factual "
        "product data. Do not use for compatibility judgements.",
        ("what are the dimensions of sku LM-12", "technical specs for this model"),
        ("product", "information"),
        ("specifications", "technical specs", "dimensions of", "what are the specs"),
        args=ProductRef,
    ),
    _r(
        "product_compatibility",
        "Say whether a product works with something the customer already owns. Use for 'will this fit' "
        "questions. Do not use for plain specification lookups.",
        ("is sku LM-12 compatible with my 2019 model", "will it fit my existing mount"),
        ("product", "information"),
        ("is it compatible", "compatible with", "will it fit", "works with my"),
        args=ProductRef,
    ),
    _r(
        "product_price_check",
        "Give the current price of a product. Use for price questions before buying. Do not use for "
        "charges already on an invoice.",
        ("how much does sku QR-14 cost", "what is the current price of this"),
        ("product", "pricing"),
        ("how much does it cost", "current price of", "what is the price", "how much does sku"),
        args=ProductRef,
    ),
    _r(
        "product_price_match",
        "Consider matching a lower price found elsewhere. Use when the customer cites a competitor. Do "
        "not use for promotional codes.",
        ("can you price match this", "it is cheaper elsewhere, will you match it"),
        ("product", "pricing"),
        ("price match", "cheaper elsewhere", "match a competitor price"),
        args=ProductRef,
    ),
    _r(
        "product_recommendation",
        "Recommend a product from the catalog based on stated needs. Use for open-ended 'what should I "
        "buy'. Do not use when the customer names a specific product.",
        ("what do you recommend for a beginner", "which one should I buy for camping"),
        ("product", "advice"),
        ("what do you recommend", "which one should i buy", "suggest a product", "recommend something"),
    ),
    _r(
        "product_review_submit",
        "Help the customer publish a product review or rating. Use for writing reviews. Do not use for "
        "complaints about a faulty item.",
        ("I want to leave a review for sku AB-99", "how do I rate the product I bought"),
        ("product", "reviews"),
        ("leave a review", "write a review", "rate the product"),
        args=ProductRef,
    ),
)


# --------------------------------------------------------------------------- #
# 10. Loyalty (7)
# --------------------------------------------------------------------------- #

_LOYALTY: tuple[Route, ...] = (
    _r(
        "loyalty_points_balance",
        "Report the customer's current loyalty point balance. Use for 'how many points do I have'. Do "
        "not use when points are missing from a specific order.",
        ("how many points do I have", "what is my loyalty points balance"),
        ("loyalty", "points"),
        ("how many points", "points balance", "loyalty points balance"),
    ),
    _r(
        "loyalty_points_redeem",
        "Spend loyalty points on a reward or a discount. Use when the customer wants to use points. Do "
        "not use to check the balance alone.",
        ("I want to redeem my points", "can I spend my points on this order"),
        ("loyalty", "points"),
        ("redeem my points", "spend my points", "use my loyalty points"),
    ),
    _r(
        "loyalty_missing_points",
        "Investigate points that were not credited for a qualifying purchase. Use when points are "
        "absent after an order. Do not use for points that have expired.",
        ("points missing from order 6612", "I did not get my points for last week's order"),
        ("loyalty", "points"),
        ("points missing", "did not get my points", "points not credited"),
        args=OrderRef,
    ),
    _r(
        "loyalty_tier_status",
        "Explain the customer's loyalty tier and what reaching the next one needs. Use for status "
        "questions. Do not use for the point balance alone.",
        ("what loyalty tier am I on", "how do I reach gold status"),
        ("loyalty", "tiers"),
        ("loyalty tier", "gold status", "membership level", "what tier am i"),
    ),
    _r(
        "loyalty_enroll",
        "Enrol the customer into the loyalty or rewards programme. Use for joining. Do not use for "
        "referral bonuses.",
        ("how do I join the loyalty programme", "sign me up for rewards"),
        ("loyalty", "membership"),
        ("join the loyalty", "sign up for rewards", "enroll in the rewards", "enrol in the rewards"),
    ),
    _r(
        "loyalty_referral_credit",
        "Handle a referral bonus that is owed or missing. Use when a friend was referred. Do not use "
        "for promotional discount codes.",
        ("I referred a friend but got no bonus", "where is my referral credit"),
        ("loyalty", "referrals"),
        ("referral bonus", "referred a friend", "referral code", "referral credit"),
    ),
    _r(
        "loyalty_expiring_points",
        "Explain when points expire and whether expiry can be deferred. Use for expiry questions. Do "
        "not use for points that were never credited.",
        ("when do my points expire", "my points are expiring this month"),
        ("loyalty", "points"),
        ("points expiring", "when do my points expire", "points expire"),
    ),
)


# --------------------------------------------------------------------------- #
# 11. Privacy & legal (7)
# --------------------------------------------------------------------------- #

_PRIVACY: tuple[Route, ...] = (
    _r(
        "privacy_data_export",
        "Handle a subject access request for a copy of the customer's personal data. Use for GDPR-style "
        "data requests. Requires the account owner. Do not use for operational CSV exports.",
        ("I want a copy of my data, gdpr request", "please send me a subject access request export"),
        ("privacy", "gdpr"),
        ("copy of my data", "gdpr request", "subject access request", "all my personal data"),
        requires=("account_owner",),
    ),
    _r(
        "privacy_data_deletion",
        "Handle a request to erase personal data. Use for right-to-be-forgotten requests. Requires the "
        "account owner. Do not use for simply closing the account.",
        ("delete my personal data", "I am exercising my right to be forgotten"),
        ("privacy", "gdpr"),
        ("delete my data", "erase my personal data", "right to be forgotten", "delete my personal data"),
        requires=("account_owner",),
    ),
    _r(
        "privacy_marketing_optout",
        "Stop marketing communications. Use for unsubscribe requests. Do not use for transactional "
        "order notifications.",
        ("unsubscribe from marketing emails", "stop sending me promotional messages"),
        ("privacy", "consent"),
        ("unsubscribe from marketing", "stop marketing emails", "opt out of marketing",
         "promotional messages"),
    ),
    _r(
        "privacy_cookie_preferences",
        "Change cookie and tracking consent. Use for cookie banner and consent questions. Do not use "
        "for marketing email preferences.",
        ("how do I change my cookie settings", "I want to withdraw cookie consent"),
        ("privacy", "consent"),
        ("cookie settings", "cookie consent", "manage cookies"),
    ),
    _r(
        "privacy_report_fraud",
        "Handle suspected fraud or unauthorised account access. Use when the customer says someone "
        "else used their account or card. Treat as urgent.",
        ("someone used my account, report fraud", "there is a fraudulent charge I did not make"),
        ("privacy", "security"),
        ("report fraud", "fraudulent charge", "someone used my account", "unauthorised access"),
    ),
    _r(
        "legal_terms_question",
        "Answer a question about the terms of service or contractual conditions. Use for legal wording "
        "questions. Do not use for the returns policy specifically.",
        ("where can I read your terms of service", "a question about your terms and conditions"),
        ("legal", "policy"),
        ("terms of service", "terms and conditions", "legal terms"),
    ),
    _r(
        "legal_accessibility_request",
        "Handle accessibility needs: alternative formats, screen-reader problems, accommodations. Use "
        "for accessibility requests. Do not use for general site bugs.",
        ("the checkout is unusable with a screen reader", "can I get this in an accessible format"),
        ("legal", "accessibility"),
        ("screen reader", "accessible format", "accessibility"),
    ),
)


# --------------------------------------------------------------------------- #
# 12. Escalation (1, pinned)
# --------------------------------------------------------------------------- #

_ESCALATION: tuple[Route, ...] = (
    _r(
        "human_handoff",
        "Escalate the conversation to a human support agent. Use when the customer asks for a person, "
        "or as the configured fallback when the router cannot commit. Pinned: always a candidate.",
        ("let me speak to a human", "I want to talk to a real agent", "get me a person please"),
        ("support", "escalation"),
        ("speak to a human", "talk to a person", "real agent", "talk to a human", "get me a person"),
        pinned=True,
        label="talking to a human agent",
    ),
)


# --------------------------------------------------------------------------- #
# The registry.
# --------------------------------------------------------------------------- #

routes: tuple[Route, ...] = (
    *_BILLING,
    *_PAYMENTS,
    *_SHIPPING,
    *_RETURNS,
    *_ORDERS,
    *_ACCOUNT,
    *_TECHNICAL,
    *_SUBSCRIPTIONS,
    *_PRODUCT,
    *_LOYALTY,
    *_PRIVACY,
    *_ESCALATION,
)
"""Every route in the catalog, in domain order."""

registry: Registry = Registry(routes)
"""The frozen catalog (plan §3.2). ``registry.version`` keys the prompt cache,
the shortlist index and every audit record."""

FALLBACK_ROUTE = "human_handoff"
"""``Router(fallback=...)``: the pinned escalation route (plan §6.6)."""

DOMAINS: tuple[str, ...] = tuple(
    dict.fromkeys(str(route.metadata["domain"]) for route in routes)
)
"""The distinct ``tags[0]`` domain labels, in catalog order."""

ENTITLEMENTS: frozenset[str] = frozenset().union(*(route.requires for route in routes))
"""Every entitlement any route asks for — hand this to ``RequestContext`` to see
the full catalog, or a subset to watch routes disappear (plan §7.1)."""


def triggers_for(route_name: str) -> tuple[str, ...]:
    """The demo stub's trigger phrases for ``route_name``.

    Fixture support only: switchboard itself never reads these. See the module
    docstring and ``demo.py`` for what the stub is and is not.
    """
    route = registry.get(route_name)
    if route is None:
        return ()
    return tuple(route.metadata.get("triggers", ()))


# --------------------------------------------------------------------------- #
# Gold cases (plan §9.1 fixture format, in-Python form).
# --------------------------------------------------------------------------- #

GoldCase = tuple[str, str]
"""``(query, expected)``. ``expected`` is a route name, :data:`CLARIFY` or
:data:`ABSTAIN`."""

CLARIFY = "clarify"
"""Expected outcome label: the query is in scope but underdetermined, so the
right answer is a question, not a guess (plan §6.4, §9.2)."""

ABSTAIN = "abstain"
"""Expected outcome label: nothing in this catalog handles the query. With
``Router(fallback="human_handoff")`` configured, a terminal abstain arrives as
``kind="route"`` with ``decision_path="fallback"`` (plan §6.6) — the demo scores
either shape as correct."""

GOLD_CASES: list[GoldCase] = [
    # -- shipping ---------------------------------------------------------- #
    ("where is my package for order 4471", "shipping_track_parcel"),
    ("when will it arrive, order 8823", "shipping_delivery_estimate"),
    ("my parcel is late for order 3390", "shipping_delayed_parcel"),
    ("the package never arrived for order 5521", "shipping_lost_parcel"),
    ("order 6640 arrived damaged, the box was crushed", "shipping_damaged_parcel"),
    ("one item missing from my order 3312", "shipping_missing_item"),
    ("can I upgrade to express for order 5540", "shipping_upgrade_speed"),
    ("held at customs, what is the import duty", "shipping_customs_fees"),
    ("do you ship to Norway", "shipping_international_availability"),
    # -- returns ----------------------------------------------------------- #
    ("I want to start a return for order 7781", "returns_start_return"),
    ("I need a return label for order 1180", "returns_print_label"),
    ("what is your return policy", "returns_policy_explain"),
    ("have you received my return for order 2004", "returns_status"),
    ("can I exchange it for another size, order 8890", "returns_exchange_item"),
    ("I want my money back for order 123", "returns_refund_request"),
    ("this is not what I ordered, wrong item in order 7712", "returns_wrong_item"),
    ("the product is defective for order 9001, it stopped working", "returns_faulty_item"),
    ("warranty claim for sku AB-99", "returns_warranty_claim"),
    # A sound route choice with an unextractable required argument: plan §3.8
    # row 4 downgrades `route` -> `clarify` and names the missing field.
    ("please refund me", CLARIFY),
    # -- orders ------------------------------------------------------------ #
    ("cancel my order 2210", "orders_cancel"),
    ("what is the status of order 9912", "orders_status"),
    ("show me my order history", "orders_history"),
    ("I want to reorder the same thing, order 3320", "orders_reorder"),
    ("add gift wrap and a gift message to order 4410", "orders_gift_wrap"),
    ("I need a bulk quote for 500 units", "orders_bulk_quote"),
    # -- billing / payments ------------------------------------------------ #
    ("can I see invoice INV-2231", "billing_invoice_view"),
    ("why was I charged 49 euros", "billing_charge_explain"),
    ("I was charged twice for the same order", "billing_duplicate_charge"),
    ("I need to update my card on file", "billing_update_payment_method"),
    ("my payment failed, the card was declined", "billing_payment_failed"),
    ("export the billing ledger for last quarter", "billing_export_ledger"),
    ("can I pay in installments", "payments_installment_plan"),
    ("what is my gift card balance", "payments_gift_card_balance"),
    ("my promo code is not working", "payments_promo_code_invalid"),
    ("where is my refund for order 4402", "payments_refund_status"),
    # -- account ----------------------------------------------------------- #
    ("I cannot log in to my account", "account_login_problem"),
    ("reset my password please", "account_password_reset"),
    ("change my email to sam@example.com", "account_change_email"),
    ("I lost my authenticator, 2fa recovery", "account_two_factor_recovery"),
    ("invite a colleague to the workspace", "account_add_user"),
    ("stop emailing me about every order", "account_notification_preferences"),
    # -- technical --------------------------------------------------------- #
    ("the app crashes on startup", "tech_app_crash"),
    ("checkout does not work", "tech_checkout_broken"),
    ("I did not receive the email confirmation", "tech_email_not_received"),
    ("rotate my api key, it was compromised", "tech_api_key_rotate"),
    ("my webhook is not firing", "tech_webhook_configure"),
    # -- subscriptions ----------------------------------------------------- #
    ("I want to upgrade my plan to pro", "subs_upgrade_plan"),
    ("cancel my subscription", "subs_cancel"),
    ("pause my subscription for two months", "subs_pause"),
    ("when does it renew", "subs_renewal_date"),
    ("turn off auto renew", "subs_auto_renew_toggle"),
    ("add 3 more seats", "subs_seats_change"),
    # -- product / loyalty / privacy --------------------------------------- #
    ("is sku QR-14 in stock", "product_availability"),
    ("notify me when sku QR-14 is back in stock", "product_restock_alert"),
    ("how many points do I have", "loyalty_points_balance"),
    ("I want to redeem my points", "loyalty_points_redeem"),
    ("points missing from order 6612", "loyalty_missing_points"),
    ("unsubscribe from marketing emails", "privacy_marketing_optout"),
    ("I want a copy of my data, gdpr request", "privacy_data_export"),
    ("someone used my account, report fraud", "privacy_report_fraud"),
    ("let me speak to a human", "human_handoff"),
    # -- genuinely ambiguous: two routes fit equally well ------------------ #
    # Each of these contains a phrase that belongs to *two* routes with
    # different consequences. Guessing is worse than asking (plan §6, rule 5).
    ("can you cancel it", CLARIFY),                       # order vs subscription
    ("I have a question about a refund", CLARIFY),        # new refund vs chasing one
    ("there is a charge on my card I do not recognise", CLARIFY),  # explain vs dispute
    ("my order is wrong", CLARIFY),                       # wrong item vs edit the order
    ("I need to change my address", CLARIFY),             # delivery vs billing address
    # -- out of scope: nothing here handles these -------------------------- #
    ("what is the capital of France", ABSTAIN),
    ("write me a poem about autumn", ABSTAIN),
    ("book me a table for two at 8pm tonight", ABSTAIN),
    ("what time does the moon rise tomorrow", ABSTAIN),
    ("translate this sentence into german", ABSTAIN),
]
"""Labelled evaluation set: 71 cases — 61 single-route, 5 ambiguous (expect a
clarify), 5 out of scope (expect an abstain or the configured fallback)."""
