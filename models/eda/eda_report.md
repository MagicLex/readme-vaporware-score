# README Vaporware — EDA

- rows: **3600**
- vaporware (archived): **1800**, alive: **1800**
- languages: 124 (top: Python, TypeScript, JavaScript, unknown, Go)

## README signal vs abandonment (correlation with label)

Positive = more of this feature -> more likely archived.

| feature | corr | vaporware mean | alive mean |
|---|---:|---:|---:|
| heading_count | -0.101 | 9.25 | 12.32 |
| readme_lines | -0.097 | 107.64 | 152.90 |
| inline_code_count | -0.097 | 10.52 | 17.75 |
| has_contributing | -0.097 | 0.24 | 0.33 |
| has_tests_mention | -0.089 | 0.46 | 0.55 |
| has_usage_section | -0.088 | 0.31 | 0.39 |
| readme_chars | -0.076 | 4874.44 | 7367.24 |
| readme_words | -0.073 | 676.78 | 1004.76 |
| has_license_section | -0.072 | 0.33 | 0.40 |
| exclamation_count | -0.066 | 3.13 | 4.39 |
| code_fence_blocks | -0.064 | 3.14 | 4.83 |
| badge_count | -0.050 | 1.51 | 2.58 |
| image_count | -0.049 | 2.68 | 4.71 |
| uppercase_word_count | -0.049 | 19.52 | 26.71 |
| has_install_section | -0.042 | 0.56 | 0.60 |
| emoji_count | -0.041 | 1.99 | 3.57 |
| link_count | -0.033 | 10.38 | 16.74 |
| rocket_count | -0.031 | 0.04 | 0.12 |
| list_item_count | -0.028 | 14.89 | 21.23 |
| badge_density | -0.024 | 0.46 | 0.52 |
| buzzword_count | -0.022 | 0.82 | 1.31 |
| sparkles_count | -0.017 | 0.04 | 0.06 |
| emoji_per_1k_chars | +0.015 | 0.54 | 0.49 |
| buzzword_per_1k_words | +0.011 | 1.19 | 1.10 |
| exclamation_per_1k_words | +0.010 | 6.75 | 6.53 |
| fire_count | -0.009 | 0.02 | 0.03 |
| wip_marker_count | +0.008 | 0.28 | 0.26 |

## Confound check: creation year by class

|   created_year |   0 |   1 |
|---------------:|----:|----:|
|           2021 | 654 | 847 |
|           2022 | 706 | 663 |
|           2023 | 440 | 290 |