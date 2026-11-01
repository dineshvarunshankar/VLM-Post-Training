import unittest
from copy import deepcopy

META = dict(fps=27.411, n_frames=411, duration_s=14.994,
            resolution=dict(width=338,height=390),
            frame_convention=dict(first_frame=1,time_s='(frame - 1) / fps',coordinates='absolute pixels, origin top-left'))

class TargetResilienceTests(unittest.TestCase):
    def test_rounded_duration_preserves_annotated_span(self):
        from scripts.train.shared.grounding import temporal_target
        cue={'time_spans':[{'start_s':0,'end_s':14.994064}]}
        target,_=temporal_target(cue,META)
        self.assertEqual(target,[{'start':'00:00','end':'00:14.994'}])

    def test_missing_duration_uses_frame_geometry(self):
        from scripts.train.shared.grounding import temporal_target
        meta=dict(META,duration_s=None)
        self.assertEqual(temporal_target({'time_spans':[{'start_s':1,'end_s':2}]},meta)[0],
                         [{'start':'00:01','end':'00:02'}])
        self.assertIsNone(meta['duration_s'])

    def test_out_of_range_span_still_fails(self):
        from scripts.train.shared.grounding import temporal_target
        with self.assertRaises(ValueError):
            temporal_target({'time_spans':[{'start_s':0,'end_s':16}]},META)

    def test_geometry_accepts_fps_rounding_but_not_mismatch(self):
        from scripts.train.shared.grounding import require_matching_geometry
        geometry=dict(width=338,height=390,n_frames=411,fps=25687500/937129)
        require_matching_geometry(META,geometry)
        for key,value in [('width',339),('n_frames',410),('fps',28)]:
            with self.assertRaises(ValueError):
                require_matching_geometry(META,dict(geometry,**{key:value}))

    def test_missing_spatial_evidence_keeps_answer_and_temporal_target(self):
        from scripts.train.shared.grounding import label_grounding_targets
        cue=dict(answer=True,evidence_min=1,evidence_slots=None,evidence_points=[],
                 time_spans=[dict(start_s=1,end_s=2)])
        issues=[]
        target,frames=label_grounding_targets(cue,META,spatial=True,temporal=True,on_issue=lambda task,error:issues.append(task))
        self.assertNotIn('evidence',target)
        self.assertEqual(target['time_spans'],[{'start':'00:01','end':'00:02'}])
        self.assertEqual(issues,['spatial'])
        self.assertTrue(frames)

    def test_negative_targets_keep_required_empty_components(self):
        from scripts.train.shared.grounding import label_grounding_targets
        target,frames=label_grounding_targets({'answer':False},META,spatial=True,temporal=True)
        self.assertEqual(target,{'evidence':[],'time_spans':[]})
        self.assertEqual(frames,[])

    def test_bad_subject_point_does_not_discard_other_frames(self):
        from scripts.train.shared.grounding import subject_targets
        meta=deepcopy(META)
        tracks={'frames':[{'frame':i,'parts':[dict(part='person:1',by='human',occluded=False,point=point)]}
                          for i,point in [(1,[150,150]),(2,[300,300])]]}
        boxes={'box_convention':'bbox_pct is percent of frame, origin top-left',
               'frames':[{'frame':i,'bbox_pct':dict(x=0,y=0,width=50,height=50)} for i in [1,2]]}
        issues=[]
        targets=subject_targets({'people':[{'id':1,'role':'subject'}]},tracks,boxes,meta,
                                on_issue=lambda error:issues.append(str(error)))
        self.assertEqual([t['frame'] for t in targets],[0])
        self.assertEqual(len(issues),1)

    def test_missing_duration_accounts_for_rounded_fps_precision(self):
        from scripts.train.shared.grounding import temporal_target
        meta=dict(META,duration_s=None,n_frames=150,fps=9.965)
        target,_=temporal_target({'time_spans':[{'start_s':0,'end_s':15.053999}]},meta)
        self.assertEqual(target,[{'start':'00:00','end':'00:15.054'}])

    def test_recovered_target_scores_full_credit_at_rounded_endpoint(self):
        from scripts.train.shared.grounding import temporal_target, temporal_duration_limit
        from scripts.train.rl.rewards import temporal_reward
        meta=dict(META,duration_s=None,n_frames=150,fps=9.965)
        spans,_=temporal_target({'time_spans':[{'start_s':0,'end_s':15.053999}]},meta)
        self.assertEqual(temporal_reward(spans,spans,temporal_duration_limit(meta)),1)

    def test_current_person_subject_types_are_counted(self):
        from scripts.train.shared.grounding import counting_targets
        points={'frames':[{'frame':1,'parts':[
            {'part':'person:1','by':'human','occluded':False,'point':[10,10]},
            {'part':'person:2','by':'human','occluded':False,'point':[20,20]}]}]}
        for subject_type in ('casualty','medic'):
            with self.subTest(subject_type=subject_type):
                counting={'subject_type':subject_type,'people':[{'id':1,'role':'subject'},{'id':2,'role':'other'}]}
                self.assertEqual(counting_targets(counting,points,META)[0]['count'],2)
        counting={'subject_type':'robot','people':[{'id':1,'role':'subject'},{'id':2,'role':'other'}]}
        self.assertEqual(counting_targets(counting,points,META)[0]['count'],1)

    def test_current_subject_requires_human_frame_visibility(self):
        from scripts.train.shared.grounding import counting_targets
        counting={'subject_type':'casualty','people':[{'id':1,'role':'subject'},{'id':2,'role':'other'}]}
        points={'frames':[{'frame':1,'parts':[{'part':'person:2','by':'human','occluded':False,'point':[20,20]}]}]}
        self.assertEqual(counting_targets(counting,points,META),[])
