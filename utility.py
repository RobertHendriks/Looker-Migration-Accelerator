“””
Enterprise Multi-Workbook Migration Orchestrator
Handles consolidation of multiple Tableau workbooks into a unified Looker project
Author: Built for Robert Hendriks
“””

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import json
from collections import defaultdict
import difflib
import hashlib

@dataclass
class UnifiedView:
“”“Represents a consolidated view across multiple workbooks”””
name: str
source_workbooks: List[str] = field(default_factory=list)
canonical_definition: Optional[str] = None
variations: List[Dict] = field(default_factory=list)
merge_strategy: str = “most_complete” # or “manual_review”

@dataclass
class UnifiedModel:
“”“Represents a consolidated Looker model”””
name: str
views: List[UnifiedView]
explores: List[Dict]
dashboards: List[str]

class ViewConsolidator:
“”“Consolidates duplicate views across workbooks”””

```
def __init__(self):
self.view_registry = {} # view_name -> list of definitions
self.similarity_threshold = 0.85

def register_view(self, view_name: str, view_definition: Dict, workbook_source: str):
"""Register a view from a workbook"""
if view_name not in self.view_registry:
self.view_registry[view_name] = []

self.view_registry[view_name].append({
'definition': view_definition,
'source': workbook_source,
'hash': self._hash_definition(view_definition)
})

def _hash_definition(self, definition: Dict) -> str:
"""Create hash of view definition for comparison"""
# Hash based on structure, not source workbook
key_elements = {
'table': definition.get('table'),
'columns': sorted([col['name'] for col in definition.get('columns', [])]),
'calc_fields': sorted([cf['name'] for cf in definition.get('calculated_fields', [])])
}
return hashlib.md5(json.dumps(key_elements, sort_keys=True).encode()).hexdigest()

def consolidate(self) -> Dict[str, UnifiedView]:
"""Consolidate all registered views"""
unified_views = {}

for view_name, definitions in self.view_registry.items():
if len(definitions) == 1:
# Only one source, no consolidation needed
unified_views[view_name] = UnifiedView(
name=view_name,
source_workbooks=[definitions[0]['source']],
canonical_definition=definitions[0]['definition']
)
else:
# Multiple sources, need to consolidate
unified = self._consolidate_view_definitions(view_name, definitions)
unified_views[view_name] = unified

return unified_views

def _consolidate_view_definitions(self, view_name: str, definitions: List[Dict]) -> UnifiedView:
"""Consolidate multiple definitions of the same view"""

# Check if all definitions are identical
hashes = [d['hash'] for d in definitions]
if len(set(hashes)) == 1:
# All identical - use first one
return UnifiedView(
name=view_name,
source_workbooks=[d['source'] for d in definitions],
canonical_definition=definitions[0]['definition'],
merge_strategy="identical"
)

# Definitions differ - need merge strategy
# Strategy: Use the most complete definition (most columns + calc fields)
most_complete = max(definitions, key=lambda d:
len(d['definition'].get('columns', [])) +
len(d['definition'].get('calculated_fields', []))
)

# Track variations for manual review
variations = []
for d in definitions:
if d['hash'] != most_complete['hash']:
variations.append({
'source': d['source'],
'difference': self._compute_difference(
most_complete['definition'],
d['definition']
)
})

return UnifiedView(
name=view_name,
source_workbooks=[d['source'] for d in definitions],
canonical_definition=most_complete['definition'],
variations=variations,
merge_strategy="most_complete" if not variations else "manual_review"
)

def _compute_difference(self, def1: Dict, def2: Dict) -> Dict:
"""Compute structural differences between two view definitions"""
cols1 = set(c['name'] for c in def1.get('columns', []))
cols2 = set(c['name'] for c in def2.get('columns', []))

calcs1 = set(c['name'] for c in def1.get('calculated_fields', []))
calcs2 = set(c['name'] for c in def2.get('calculated_fields', []))

return {
'columns_only_in_def1': list(cols1 - cols2),
'columns_only_in_def2': list(cols2 - cols1),
'calc_fields_only_in_def1': list(calcs1 - calcs2),
'calc_fields_only_in_def2': list(calcs2 - calcs1)
}
```

class CalculatedFieldConsolidator:
“”“Consolidates duplicate calculated fields across workbooks”””

```
def __init__(self):
self.field_registry = defaultdict(list)

def register_field(self, field_name: str, formula: str, workbook_source: str):
"""Register a calculated field"""
normalized_formula = self._normalize_formula(formula)

self.field_registry[field_name].append({
'formula': formula,
'normalized': normalized_formula,
'source': workbook_source
})

def _normalize_formula(self, formula: str) -> str:
"""Normalize formula for comparison (remove whitespace, lowercase)"""
import re
normalized = re.sub(r'\s+', ' ', formula.lower().strip())
return normalized

def find_duplicates(self) -> Dict[str, List[Dict]]:
"""Find calculated fields with same name but different formulas"""
duplicates = {}

for field_name, definitions in self.field_registry.items():
if len(definitions) <= 1:
continue

# Group by normalized formula
formula_groups = defaultdict(list)
for d in definitions:
formula_groups[d['normalized']].append(d)

if len(formula_groups) > 1:
# Same field name, different formulas!
duplicates[field_name] = {
'variations': list(formula_groups.values()),
'recommendation': self._recommend_canonical(formula_groups)
}

return duplicates

def _recommend_canonical(self, formula_groups: Dict) -> Dict:
"""Recommend which formula to use as canonical"""
# Use the most common formula
most_common = max(formula_groups.items(), key=lambda x: len(x[1]))

return {
'formula': most_common[1][0]['formula'],
'sources': [d['source'] for d in most_common[1]],
'reason': f'Most common ({len(most_common[1])} occurrences)'
}
```

class EnterpriseMigrationOrchestrator:
“”“Orchestrates migration of multiple Tableau workbooks into unified Looker project”””

```
def __init__(self, workbook_paths: List[str]):
self.workbook_paths = [Path(p) for p in workbook_paths]
self.parsers = []
self.view_consolidator = ViewConsolidator()
self.calc_consolidator = CalculatedFieldConsolidator()
self.unified_views = {}
self.duplicate_calcs = {}

def analyze_all_workbooks(self) -> Dict:
"""Parse and analyze all workbooks"""
print(f"Analyzing {len(self.workbook_paths)} workbooks...")

# Import here to avoid circular dependency
from tableau_looker_migrator import TableauParser

# Parse each workbook
for twb_path in self.workbook_paths:
print(f"\n📊 Parsing: {twb_path.name}")
parser = TableauParser(str(twb_path))
parser.parse()
self.parsers.append(parser)

# Register views
for ds in parser.data_sources:
view_def = {
'name': ds.name,
'table': ds.table,
'database': ds.database,
'schema': ds.schema,
'columns': ds.columns,
'calculated_fields': [
{'name': cf.name, 'formula': cf.formula}
for cf in ds.calculated_fields
]
}
self.view_consolidator.register_view(
ds.name,
view_def,
twb_path.name
)

# Register calculated fields
for cf in ds.calculated_fields:
self.calc_consolidator.register_field(
cf.name,
cf.formula,
twb_path.name
)

# Consolidate
print("\n🔄 Consolidating views...")
self.unified_views = self.view_consolidator.consolidate()

print("\n🔍 Finding duplicate calculations...")
self.duplicate_calcs = self.calc_consolidator.find_duplicates()

return self._generate_consolidation_report()

def _generate_consolidation_report(self) -> Dict:
"""Generate report on consolidation findings"""
total_views_before = sum(len(p.data_sources) for p in self.parsers)
total_views_after = len(self.unified_views)

views_requiring_review = [
v for v in self.unified_views.values()
if v.merge_strategy == "manual_review"
]

report = {
'summary': {
'workbooks_analyzed': len(self.workbook_paths),
'views_before_consolidation': total_views_before,
'views_after_consolidation': total_views_after,
'views_eliminated': total_views_before - total_views_after,
'views_requiring_manual_review': len(views_requiring_review),
'duplicate_calculated_fields': len(self.duplicate_calcs)
},
'unified_views': {
name: {
'sources': view.source_workbooks,
'merge_strategy': view.merge_strategy,
'variations': len(view.variations)
}
for name, view in self.unified_views.items()
},
'views_requiring_review': [
{
'name': v.name,
'sources': v.source_workbooks,
'variations': v.variations
}
for v in views_requiring_review
],
'duplicate_calculated_fields': {
name: {
'variation_count': len(data['variations']),
'recommended_formula': data['recommendation']['formula'][:100] + '...',
'sources': data['recommendation']['sources']
}
for name, data in self.duplicate_calcs.items()
}
}

return report

def generate_unified_lookml(self) -> Dict[str, str]:
"""Generate consolidated LookML project"""
lookml_files = {}

# Single unified model
lookml_files['models/enterprise.model.lkml'] = self._generate_unified_model()

# Consolidated views (one per unique view)
for view_name, unified_view in self.unified_views.items():
view_lkml = self._generate_unified_view_lkml(unified_view)
lookml_files[f'views/{self._sanitize_name(view_name)}.view.lkml'] = view_lkml

# Explores organized by domain
explores_by_domain = self._organize_explores_by_domain()
for domain, explores in explores_by_domain.items():
explore_lkml = self._generate_domain_explores(domain, explores)
lookml_files[f'explores/{domain}_explores.lkml'] = explore_lkml

# Dashboards (one per original workbook)
for parser in self.parsers:
for dashboard in parser.dashboards:
dash_lkml = self._generate_dashboard_lkml(dashboard, parser)
dash_name = self._sanitize_name(dashboard.name)
lookml_files[f'dashboards/{dash_name}.dashboard.lookml'] = dash_lkml

# Governance review file
lookml_files['GOVERNANCE_REVIEW.md'] = self._generate_governance_review()

return lookml_files

def _generate_unified_model(self) -> str:
"""Generate single unified model file"""
lkml = """# Enterprise Unified Data Model
```

# Consolidated from multiple Tableau workbooks

connection: “enterprise_database”

# Include all views

include: “/views/*.view.lkml”

# Include explores organized by domain

include: “/explores/*.lkml”

# Datagroups

datagroup: default_datagroup {
sql_trigger: SELECT MAX(updated_at) FROM etl_metadata ;;
max_cache_age: “1 hour”
}

persist_with: default_datagroup

# Access grants

access_grant: finance_only {
user_attribute: department
allowed_values: [“finance”, “executive”]
}

access_grant: sales_only {
user_attribute: department
allowed_values: [“sales”, “executive”]
}
“””
return lkml

```
def _generate_unified_view_lkml(self, unified_view: UnifiedView) -> str:
"""Generate LookML for a consolidated view"""
view_name = self._sanitize_name(unified_view.name)
definition = unified_view.canonical_definition

lkml = f"""# Unified View: {unified_view.name}
```

# Consolidated from: {’, ’.join(unified_view.source_workbooks)}

“””

```
if unified_view.merge_strategy == "manual_review":
lkml += """# ⚠️ MANUAL REVIEW REQUIRED
```

# This view has variations across workbooks

# See GOVERNANCE_REVIEW.md for details

“””

```
lkml += f"""view: {view_name} {{
```

sql_table_name: “””

```
if definition.get('schema') and definition.get('table'):
lkml += f"`{definition.get('database')}.{definition.get('schema')}.{definition.get('table')}` ;;\n"
else:
lkml += f"`{definition.get('table', view_name)}` ;;\n"

lkml += "\n # Dimensions\n"
for col in definition.get('columns', [])[:10]: # Limit for example
col_name = self._sanitize_name(col['name'])
lkml += f""" dimension: {col_name} {{
type: {self._map_datatype(col.get('datatype', 'string'))}
sql: ${{TABLE}}.{col['name']} ;;
```

}}

“””

```
lkml += " # Measures\n"
for calc in definition.get('calculated_fields', []):
if calc.get('role') == 'measure':
measure_name = self._sanitize_name(calc['name'])
lkml += f""" measure: {measure_name} {{
type: number
# TODO: Review formula conversion
# Original: {calc['formula'][:50]}...
sql: ... ;;
```

}}

“””

```
lkml += "}\n"
return lkml

def _organize_explores_by_domain(self) -> Dict[str, List]:
"""Organize explores by business domain"""
# Simple heuristic: group by view name patterns
domains = defaultdict(list)

for view_name in self.unified_views.keys():
# Detect domain from view name
if any(word in view_name.lower() for word in ['sale', 'order', 'revenue']):
domain = 'sales'
elif any(word in view_name.lower() for word in ['customer', 'account']):
domain = 'customer'
elif any(word in view_name.lower() for word in ['product', 'inventory']):
domain = 'product'
elif any(word in view_name.lower() for word in ['finance', 'ledger', 'transaction']):
domain = 'finance'
else:
domain = 'general'

domains[domain].append(view_name)

return domains

def _generate_domain_explores(self, domain: str, view_names: List[str]) -> str:
"""Generate explores for a business domain"""
lkml = f"""# {domain.title()} Domain Explores
```

“””

```
for view_name in view_names:
sanitized = self._sanitize_name(view_name)
lkml += f"""explore: {sanitized} {{
```

label: “{view_name}”
group_label: “{domain.title()}”

# TODO: Add joins to related views

# TODO: Configure access grants if needed

}}

“””

```
return lkml

def _generate_dashboard_lkml(self, dashboard, parser) -> str:
"""Generate dashboard LookML (simplified)"""
dash_name = self._sanitize_name(dashboard.name)

lkml = f"""- dashboard: {dash_name}
```

title: {dashboard.name}
layout: newspaper
description: “Migrated from {parser.file_path.name}”

elements:

# TODO: Add dashboard elements

# Original had {len(dashboard.worksheets)} worksheets

“””
return lkml

```
def _generate_governance_review(self) -> str:
"""Generate governance review document"""
md = """# Governance Review Required
```

## Views Requiring Manual Review

“””

```
for view in self.unified_views.values():
if view.merge_strategy == "manual_review":
md += f"\n### {view.name}\n\n"
md += f"**Found in workbooks:** {', '.join(view.source_workbooks)}\n\n"
md += "**Variations:**\n"
for var in view.variations:
md += f"- Source: `{var['source']}`\n"
if var['difference']['columns_only_in_def2']:
md += f" - Extra columns: {', '.join(var['difference']['columns_only_in_def2'])}\n"
if var['difference']['calc_fields_only_in_def2']:
md += f" - Extra calculations: {', '.join(var['difference']['calc_fields_only_in_def2'])}\n"
md += "\n"

md += "\n## Duplicate Calculated Fields\n\n"

for field_name, data in self.duplicate_calcs.items():
md += f"\n### {field_name}\n\n"
md += f"**{len(data['variations'])} different definitions found**\n\n"
md += f"**Recommended canonical definition:**\n"
md += f"```\n{data['recommendation']['formula']}\n```\n"
md += f"Used in: {', '.join(data['recommendation']['sources'])}\n\n"

return md

def _sanitize_name(self, name: str) -> str:
"""Sanitize name for LookML"""
import re
sanitized = re.sub(r'[^\w\s]', '', name)
sanitized = sanitized.lower().replace(' ', '_')
return re.sub(r'_+', '_', sanitized).strip('_')

def _map_datatype(self, tableau_type: str) -> str:
"""Map datatypes"""
mapping = {
'string': 'string',
'integer': 'number',
'real': 'number',
'boolean': 'yesno',
'date': 'date',
'datetime': 'time'
}
return mapping.get(tableau_type.lower(), 'string')

def export_unified_project(self, output_dir: str = "./enterprise_migration"):
"""Export complete unified Looker project"""
output_path = Path(output_dir)
output_path.mkdir(exist_ok=True)

# Generate consolidation report
report = self.analyze_all_workbooks()

report_file = output_path / "consolidation_report.json"
with open(report_file, 'w') as f:
json.dump(report, f, indent=2)
print(f"\n✓ Consolidation report: {report_file}")

# Generate unified LookML
print("\n🔄 Generating unified LookML project...")
lookml_files = self.generate_unified_lookml()

# Create directory structure
(output_path / "lookml" / "models").mkdir(parents=True, exist_ok=True)
(output_path / "lookml" / "views").mkdir(parents=True, exist_ok=True)
(output_path / "lookml" / "explores").mkdir(parents=True, exist_ok=True)
(output_path / "lookml" / "dashboards").mkdir(parents=True, exist_ok=True)

# Write all files
for filename, content in lookml_files.items():
file_path = output_path / "lookml" / filename
file_path.parent.mkdir(parents=True, exist_ok=True)
with open(file_path, 'w') as f:
f.write(content)
print(f"✓ Generated: {file_path}")

# Print summary
print(f"\n{'='*60}")
print("🎉 Enterprise Migration Complete!")
print(f"{'='*60}")
print(f"\n📊 Consolidation Results:")
print(f" Workbooks analyzed: {report['summary']['workbooks_analyzed']}")
print(f" Views before: {report['summary']['views_before_consolidation']}")
print(f" Views after: {report['summary']['views_after_consolidation']}")
print(f" Eliminated: {report['summary']['views_eliminated']} duplicate views")
print(f" ⚠️ Manual review needed: {report['summary']['views_requiring_manual_review']} views")
print(f" ⚠️ Duplicate calcs found: {report['summary']['duplicate_calculated_fields']}")
print(f"\n📁 Output: {output_path}/lookml/")
print(f"\n⚠️ IMPORTANT: Review GOVERNANCE_REVIEW.md before deploying!")
```

# Example usage

if **name** == “**main**”:
import sys

```
print("="*60)
print("Enterprise Multi-Workbook Migration Orchestrator")
print("="*60)

if len(sys.argv) < 2:
print("\nUsage: python enterprise_migrator.py <twb_file1> <twb_file2> ...")
print("\nExample:")
print(" python enterprise_migrator.py sales.twb marketing.twb finance.twb")
sys.exit(1)

workbook_files = sys.argv[1:]

print(f"\n📊 Processing {len(workbook_files)} workbooks")
print("="*60)

orchestrator = EnterpriseMigrationOrchestrator(workbook_files)
orchestrator.export_unified_project()
```
