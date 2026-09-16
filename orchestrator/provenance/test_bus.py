"""Message parsing, addressing and delivery."""
import tempfile
import unittest
from pathlib import Path

from orchestrator.provenance.bus import Bus, parse, render

RECIPIENTS = {"agent-1", "agent-2"}


class Parsing(unittest.TestCase):
    def test_ids_continue_from_the_bus(self):
        messages = parse('<msg to="agent-1" kind="query">a</msg>'
                         '<msg to="all" kind="assertion">b</msg>',
                         "planner", 3, RECIPIENTS, start_id=7)
        self.assertEqual([m["id"] for m in messages], ["m7", "m8"])
        self.assertEqual([m["step"] for m in messages], [3, 3])

    def test_private_text_outside_blocks_is_not_published(self):
        messages = parse('thinking out loud <msg to="agent-1" kind="query">a</msg> more thoughts',
                         "planner", 0, RECIPIENTS)
        self.assertEqual([m["text"] for m in messages], ["a"])

    def test_malformed_blocks_raise_instead_of_being_dropped(self):
        for bad in ('<msg kind="query">a</msg>',
                    '<msg to="agent-1" kind="gossip">a</msg>',
                    '<msg to="ghost" kind="query">a</msg>',
                    '<msg to="agent-1" kind="query">   </msg>',
                    'dangling <msg to="agent-1" kind="query">a'):
            with self.assertRaises(ValueError, msg=bad):
                parse(bad, "planner", 0, RECIPIENTS)

    def test_oversized_body_is_truncated_not_rejected(self):
        messages = parse(f'<msg to="agent-1" kind="query">{"x" * 9000}</msg>',
                         "planner", 0, RECIPIENTS)
        self.assertEqual(len(messages[0]["text"]), 4000)


class Delivery(unittest.TestCase):
    def setUp(self):
        self.bus = Bus()
        self.bus.extend(parse('<msg to="agent-1" kind="query">for one</msg>'
                              '<msg to="agent-2" kind="query">for two</msg>'
                              '<msg to="all" kind="assertion">for everyone</msg>',
                              "planner", 0, RECIPIENTS, start_id=0))

    def test_an_agent_sees_only_its_own_mail_and_broadcasts(self):
        self.assertEqual([m["text"] for m in self.bus.visible("agent-1")],
                         ["for one", "for everyone"])
        self.assertEqual([m["text"] for m in self.bus.visible("agent-2")],
                         ["for two", "for everyone"])

    def test_a_sender_is_not_shown_its_own_messages(self):
        self.assertEqual(self.bus.visible("planner"), [])

    def test_delivery_does_not_repeat_what_was_already_seen(self):
        seen = set()
        first = self.bus.delivered("agent-1", seen)
        seen.update(m["id"] for m in first)
        self.assertEqual(len(first), 2)
        self.assertEqual(self.bus.delivered("agent-1", seen), [])

    def test_render_names_sender_recipient_and_kind(self):
        text = render(self.bus.visible("agent-1"))
        self.assertIn("planner to agent-1 (query)", text)
        self.assertIn("planner to everyone (assertion)", text)
        self.assertEqual(render([]), "No new messages.")

    def test_round_trip_through_disk_preserves_order_and_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bus.jsonl"
            self.bus.dump(path)
            self.assertEqual(Bus.load(path).messages, self.bus.messages)

    def test_copy_is_independent(self):
        clone = self.bus.copy()
        clone.messages[0]["text"] = "edited"
        self.assertEqual(self.bus.messages[0]["text"], "for one")


if __name__ == "__main__":
    unittest.main()
