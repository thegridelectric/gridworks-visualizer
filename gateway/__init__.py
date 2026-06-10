"""Centralized realtime gateway.

Consumes all SCADA telemetry from the GridWorks RabbitMQ broker (AMQP) and
fans it out to dashboard WebSocket clients, replacing the per-house webinter
processes. Read-only: it never publishes to the broker.
"""
