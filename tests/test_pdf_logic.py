import sys
import os
from pathlib import Path
import pytest
import datetime
from unittest.mock import MagicMock

# --- Mock DB layer before imports ---
db_mock = MagicMock()
db_mock.AsyncSessionLocal = MagicMock()
db_mock.Base = MagicMock
sys.modules["app.db"] = MagicMock()
sys.modules["app.db.database"] = db_mock
# ------------------------------------

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from app.pdf.service import PDFGenerator, PDFLayoutConfig

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def mock_assets(temp_dir):
    # Create dummy templates
    base_pdf = temp_dir / "base_template.pdf"
    cover_pdf = temp_dir / "cover_template.pdf"
    
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Base Template")
    doc.save(base_pdf)
    doc.close()
    
    doc2 = fitz.open()
    page2 = doc2.new_page()
    page2.insert_text((50, 50), "Cover Template")
    doc2.save(cover_pdf)
    doc2.close()
    
    return base_pdf, cover_pdf

def test_pdf_generation(temp_dir, mock_assets):
    base_template, cover_template = mock_assets
    
    generator = PDFGenerator(str(temp_dir))
    
    # Override font path for test if needed or ensure it fails gracefully or uses fallback
    # PDFGenerator uses LAYOUT.FONT_PATH
    # We can mock LAYOUT or ensure PyMuPDF fallback works if font not found
    
    export_info = {
        "cover_title": "Test Report",
        "cover_company_name": "Test Company",
        "label_config": {"visible_keys": ["일자"]}
    }
    
    clusters = [
        {
            "name": "Cluster 1",
            "photos": [
                {
                    "local_path": str(base_template), # Use a valid file path (can be image or pdf, generic)
                    # wait, insert_image expects image. PDF might fail?
                    # Let's create a dummy image
                    "db_labels": {"일자": "2023.01.01"},
                    "timestamp": None
                }
            ]
        }
    ]
    
    # Create dummy image
    img_path = temp_dir / "test.jpg"
    # Create valid minimal jpg
    with open(img_path, "wb") as f:
         f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xc0\x00\x11\x08\x00\n\x00\n\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\n\x0b\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xbf\x00')
    
    clusters[0]["photos"][0]["local_path"] = str(img_path)

    output_path = generator.generate(
        export_info=export_info,
        clusters=clusters,
        base_template_path=base_template,
        cover_template_path=cover_template
    )
    
    assert os.path.exists(output_path)
    assert output_path.endswith(".pdf")
    print(f"Generated PDF at: {output_path}")
