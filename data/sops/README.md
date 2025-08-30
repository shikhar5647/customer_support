# SOPs (Standard Operating Procedures) Directory

This directory contains Standard Operating Procedure documents for different business domains.

## Structure

- `ecommerce/` - E-commerce related SOPs
- `telecom/` - Telecommunications related SOPs  
- `utilities/` - Utility services related SOPs
- `general/` - General customer service SOPs

## File Formats

Supported formats:
- JSON (.json)
- YAML (.yml, .yaml)
- Text (.txt)
- Markdown (.md)

## Adding New SOPs

1. Create a new file in the appropriate domain directory
2. Follow the SOP document schema:
   ```json
   {
     "sop_id": "UNIQUE_ID",
     "title": "SOP Title",
     "domain": "domain_name",
     "version": "1.0",
     "content": "SOP content here...",
     "sections": [],
     "metadata": {},
     "tags": ["tag1", "tag2"]
   }
   ```

3. The system will automatically load and index new SOPs

## Sample SOPs

The system includes sample SOPs for demonstration purposes. In production, replace these with your actual business procedures.
