"""Review flags report provenance references without policing ordinary words."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality import review_response

class QualityTests(unittest.TestCase):
    def test_clean_text_does_not_claim_review_or_visual_approval(self):
        result = review_response({'answer': 'A person sits on the grass.'})
        self.assertNotIn('needs_review', result)
        self.assertEqual(result['source_reference_check'], 'no_flags')
        flagged = review_response({'answer': 'The annotator says a person sits.'})
        self.assertEqual(flagged['source_reference_check'], 'flagged')

    def test_explicit_references_flagged_but_not_ordinary_words_or_markup(self):
        result = review_response({'reasoning_process': 'The annotator says the subject moves. <evidence><cue>ground_truth_label</cue><point>[1,2]</point><time>00:01</time></evidence>', 'answer': {'label': 1}})
        self.assertEqual(len(result['flags']), 1)
        self.assertEqual(result['flags'][0]['field'], 'reasoning_process')
        self.assertIn('annotator says', result['flags'][0]['match'].lower())
        self.assertEqual(result['text_statistics']['reasoning_process']['prose_words'], 6)
        clean = review_response({'answer': 'A label is visible on a tracked vehicle.'})
        self.assertEqual(clean['flags'], [])
        self.assertEqual(clean['text_statistics']['answer']['characters'], 40)

    def test_flags_do_not_rewrite_response(self):
        response = {'answer': 'According to the transcript, a person moves.'}
        original = dict(response)
        result = review_response(response)
        self.assertTrue(result['flags'])
        self.assertEqual(response, original)
