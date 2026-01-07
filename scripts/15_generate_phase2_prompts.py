# scripts/15_generate_phase2_prompts.py
"""
Phase 2: Hypothesis-Driven Prompt Generation
Creates 9 new .txt files in prompts/ (just like Phase 1)

Run: python scripts/15_generate_phase2_prompts.py
"""

from pathlib import Path
from config import PROMPTS_DIR, TABLES_DIR
import pandas as pd

# =============================================================================
# PHASE 2 PROMPTS (CORRECTED - functional versions)
# =============================================================================

PHASE2_PROMPTS = {
    
    # =========================================================================
    # MINIMAL STRATEGY (v4-v6) - Tests H1, H2, H3
    # =========================================================================
    
    'minimal_v4': {
        'hypothesis': 'H1: Ultra-simple prompts MORE robust',
        'prediction': 'SD < 0.163',
        'text': '''Classify the CEFR level of this essay as A2, B1, B2, C1, or C2.

{essay_text}

CEFR Level:'''
    },
    
    'minimal_v5': {
        'hypothesis': 'H2: Length-aware improves long essays',
        'prediction': 'Long accuracy > 6.4%',
        'text': '''Classify this essay's CEFR level (A2, B1, B2, C1, or C2). Consider essay length as a proficiency indicator.

Essay: {essay_text}

CEFR Level:'''
    },
    
    'minimal_v6': {
        'hypothesis': 'H3: Ordinal constraints reduce errors',
        'prediction': 'Off-by-2+ < 10%',
        'text': '''Classify this essay's CEFR level: A2, B1, B2, C1, or C2.
Note: These levels are ordered from beginner (A2) to proficient (C2). Avoid large jumps.

Essay: {essay_text}

CEFR Level:'''
    },
    
    # =========================================================================
    # RUBRIC STRATEGY (v4-v6) - Tests H4, H5, H6
    # =========================================================================
    
    'rubric_v4': {
        'hypothesis': 'H4: Examples increase accuracy',
        'prediction': 'Accuracy > 35.6%',
        'text': '''Classify this essay's CEFR level using these criteria:

A2: Simple sentences, basic vocabulary, frequent errors (50-100 words typical)
B1: Some complex structures, everyday vocabulary, some errors (100-150 words typical)
B2: Generally accurate, varied structures, good vocabulary (150-250 words typical)
C1: Sophisticated structures, wide vocabulary, rare errors (250+ words typical)
C2: Near-native fluency, precise language, virtually error-free (250+ words typical)

Example A2: "I like play football. My friend is good."
Example B2: "Although football requires skill, I enjoy playing it because it promotes teamwork."
Example C2: "The intricacies of the sport notwithstanding, I find it intellectually stimulating."

Essay: {essay_text}

CEFR Level:'''
    },
    
    'rubric_v5': {
        'hypothesis': 'H5: Conservative strategy identifies uncertain cases',
        'prediction': 'High-confidence accuracy > 50%',
        'text': '''Classify this essay's CEFR level using these criteria:

A2: Simple sentences, basic vocabulary, frequent errors
B1: Some complex structures, everyday vocabulary, some errors
B2: Generally accurate, varied structures, good vocabulary
C1: Sophisticated structures, wide vocabulary, rare errors
C2: Near-native fluency, precise language, virtually error-free

Essay: {essay_text}

Provide ONLY the CEFR level. If you are uncertain between two adjacent levels, output the lower level.'''
    },
    
    'rubric_v6': {
        'hypothesis': 'H6: Adjacent-awareness increases adjacent accuracy',
        'prediction': 'Adjacent accuracy > 68.9%',
        'text': '''Classify this essay's CEFR level using these criteria:

A2: Simple sentences, basic vocabulary, frequent errors
B1: Some complex structures, everyday vocabulary, some errors (Note: Often overlaps with A2 and B2)
B2: Generally accurate, varied structures, good vocabulary (Note: Often overlaps with B1 and C1)
C1: Sophisticated structures, wide vocabulary, rare errors (Note: Often overlaps with B2 and C2)
C2: Near-native fluency, precise language, virtually error-free

Note: Adjacent levels often overlap. Choose the level that best fits overall, knowing perfect classification is difficult.

Essay: {essay_text}

CEFR Level:'''
    },
    
    # =========================================================================
    # COT STRATEGY (v4-v6) - Tests H7, H8, H9
    # =========================================================================
    
    'cot_v4': {
        'hypothesis': 'H7: Structured CoT reduces variance',
        'prediction': 'SD < 0.205',
        'text': '''Classify this essay's CEFR level through structured analysis.

Essay: {essay_text}

Follow these exact steps:
Step 1: Count sentences (1-3 = likely A2/B1; 4-8 = likely B2/C1; 9+ = likely C1/C2)
Step 2: Identify complex structures (subordination, passive voice, participles)
Step 3: Assess vocabulary sophistication (basic/everyday/academic/specialized)
Step 4: Determine CEFR level based on Steps 1-3

CEFR Level:'''
    },
    
    'cot_v5': {
        'hypothesis': 'H8: Constraints prevent severe errors',
        'prediction': 'Off-by-2+ < Phase 1 CoT',
        'text': '''Classify this essay's CEFR level through careful reasoning.

Essay: {essay_text}

Analyze: vocabulary, grammar, coherence, and fluency.

Important constraints:
- Very short essays (<50 words) are typically A2-B1, rarely C1-C2
- Very long essays (>250 words) are typically B2-C2, rarely A2
- Adjacent levels (e.g., B1 and B2) are acceptable; distant levels (e.g., A2 and C1) are unlikely

After analysis, what is the CEFR level?

CEFR Level:'''
    },
    
    'cot_v6': {
        'hypothesis': 'H9: QWK-awareness improves ordinal agreement',
        'prediction': 'QWK > 0.218',
        'text': '''Classify this essay's CEFR level using ordinal reasoning.

Essay: {essay_text}

CEFR levels are ordered: A2 < B1 < B2 < C1 < C2

Analyze the essay and determine its level. Remember:
- Being off by 1 level (e.g., predicting B1 when true level is B2) is acceptable
- Being off by 2+ levels (e.g., predicting A2 when true level is B2) is a serious error

Consider: Is this essay closer to beginner (A2) or proficient (C2)? Then refine to exact level.

CEFR Level:'''
    }
}

# Hypothesis metadata for tracking
HYPOTHESES_METADATA = {
    'H1': {
        'hypothesis': 'Ultra-simple prompts (18 words) MORE robust than Phase 1 minimal (31 words)',
        'baseline': 'minimal_v1 (SD=0.163)',
        'test_prompt': 'minimal_v4',
        'metric': 'robustness_sd',
        'prediction': 'Phase2 < 0.163'
    },
    'H2': {
        'hypothesis': 'Length-aware instructions IMPROVE long-essay accuracy',
        'baseline': 'minimal_v1 (long=6.4%)',
        'test_prompt': 'minimal_v5',
        'metric': 'accuracy_long',
        'prediction': 'Phase2 > 6.4%'
    },
    'H3': {
        'hypothesis': 'Ordinal constraints REDUCE off-by-2+ errors',
        'baseline': 'minimal_v1 (~10%)',
        'test_prompt': 'minimal_v6',
        'metric': 'off_by_2_plus_pct',
        'prediction': 'Phase2 < 10%'
    },
    'H4': {
        'hypothesis': 'Rubric with examples INCREASES overall accuracy',
        'baseline': 'rubric_v1 (35.6%)',
        'test_prompt': 'rubric_v4',
        'metric': 'exact_accuracy',
        'prediction': 'Phase2 > 35.6%'
    },
    'H5': {
        'hypothesis': 'Conservative strategy IDENTIFIES uncertain predictions',
        'baseline': 'rubric_v1',
        'test_prompt': 'rubric_v5',
        'metric': 'confidence_separation',
        'prediction': 'High-conf acc > 50%'
    },
    'H6': {
        'hypothesis': 'Adjacent-level awareness INCREASES adjacent accuracy',
        'baseline': 'rubric_v1 (68.9%)',
        'test_prompt': 'rubric_v6',
        'metric': 'adjacent_accuracy',
        'prediction': 'Phase2 > 68.9%'
    },
    'H7': {
        'hypothesis': 'Structured 4-step CoT MORE robust than free-form',
        'baseline': 'cot_v1 (SD=0.205)',
        'test_prompt': 'cot_v4',
        'metric': 'robustness_sd',
        'prediction': 'Phase2 < 0.205'
    },
    'H8': {
        'hypothesis': 'CoT with constraints PREVENTS severe errors',
        'baseline': 'cot_v1',
        'test_prompt': 'cot_v5',
        'metric': 'off_by_2_plus_pct',
        'prediction': 'Phase2 < Phase1 CoT'
    },
    'H9': {
        'hypothesis': 'QWK-aware CoT IMPROVES ordinal agreement',
        'baseline': 'cot_v1 (QWK=0.218)',
        'test_prompt': 'cot_v6',
        'metric': 'qwk',
        'prediction': 'Phase2 > 0.218'
    }
}

# =============================================================================
# GENERATE .TXT FILES
# =============================================================================

def generate_prompt_files():
    """Generate Phase 2 prompt .txt files (just like Phase 1)"""
    
    print("="*70)
    print("PHASE 2: GENERATING PROMPT FILES")
    print("="*70)
    
    # Create directory
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating 9 prompt files in: {PROMPTS_DIR}/")
    print()
    
    for prompt_name, prompt_data in PHASE2_PROMPTS.items():
        
        # Write .txt file
        prompt_file = PROMPTS_DIR / f"{prompt_name}.txt"
        prompt_file.write_text(prompt_data['text'])
        
        # Show info
        print(f"✓ {prompt_name}.txt")
        print(f"  {prompt_data['hypothesis']}")
        print(f"  Prediction: {prompt_data['prediction']}")
        print()
    
    print("="*70)
    print(f"✓ Generated {len(PHASE2_PROMPTS)} prompt files!")
    print("="*70)

# =============================================================================
# CREATE HYPOTHESIS TRACKING TABLE
# =============================================================================

def create_hypothesis_table():
    """Create hypothesis tracking CSV"""
    
    print("\nCreating hypothesis tracking table...")
    
    rows = []
    for h_id, h_data in HYPOTHESES_METADATA.items():
        rows.append({
            'hypothesis_id': h_id,
            'hypothesis': h_data['hypothesis'],
            'baseline_prompt': h_data['baseline'],
            'test_prompt': h_data['test_prompt'],
            'metric': h_data['metric'],
            'prediction': h_data['prediction'],
            'phase1_result': 'TBD',
            'phase2_result': 'TBD',
            'hypothesis_supported': 'TBD'
        })
    
    df = pd.DataFrame(rows)
    output_file = TABLES_DIR / "phase2_hypotheses.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved: {output_file}")
    
    # Print table
    print("\n" + "="*70)
    print("HYPOTHESES SUMMARY")
    print("="*70)
    print()
    print(df[['hypothesis_id', 'hypothesis', 'prediction']].to_string(index=False))

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all Phase 2 files"""
    
    generate_prompt_files()
    create_hypothesis_table()
    
    print("\n" + "="*70)
    print("✓ PHASE 2 SETUP COMPLETE!")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Review prompts:")
    print(f"   ls {PROMPTS_DIR}/*.txt")
    print("   # Should see: minimal_v1-v6, rubric_v1-v6, cot_v1-v6")
    print()
    print("2. Run Phase 2 experiment:")
    print("   python scripts/08_run_experiment.py --phase 2")
    print()
    print("3. Compare phases:")
    print("   python scripts/17_compare_phases_and_test_hypotheses.py")

if __name__ == "__main__":
    main()
