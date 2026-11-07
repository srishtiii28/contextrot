import os
from pathlib import Path
from sqlalchemy import select, delete
from app.database.database import SessionLocal
from app.database.models import KnowledgeBase, Document, DocumentChunk, DocumentImage
from app.services.rag_service import ingest_pdf

db = SessionLocal()

# 1. Clean up demo KBs
for kb_id in [2, 3]:
    kb = db.get(KnowledgeBase, kb_id)
    if kb:
        # Delete docs
        docs = list(db.scalars(select(Document).where(Document.knowledge_base_id == kb_id)))
        for d in docs:
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == d.id))
            db.execute(delete(DocumentImage).where(DocumentImage.document_id == d.id))
            db.delete(d)
        db.delete(kb)
db.commit()

# 2. Configure the 4 core libraries
libraries_spec = [
    {"id": 1, "name": "Biology", "subject": "Biology"},
    {"id": 4, "name": "Maths", "subject": "Mathematics"},
    {"id": 5, "name": "Physics", "subject": "Physics"},
    {"id": 6, "name": "Chemistry", "subject": "Chemistry"},
]

for spec in libraries_spec:
    kb = db.get(KnowledgeBase, spec["id"])
    if kb:
        kb.name = spec["name"]
        kb.subject = spec["subject"]
    else:
        kb = KnowledgeBase(id=spec["id"], name=spec["name"], subject=spec["subject"], owner_id=13)
        db.add(kb)
db.commit()

# 3. Clean duplicate document in Biology (ID 5 if duplicate of ID 4)
bio_docs = list(db.scalars(select(Document).where(Document.knowledge_base_id == 1).order_by(Document.id)))
if len(bio_docs) > 1:
    # keep first, remove second
    d_del = bio_docs[1]
    db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == d_del.id))
    db.execute(delete(DocumentImage).where(DocumentImage.document_id == d_del.id))
    db.delete(d_del)
    db.commit()

# 4. Check Chemistry Part 1
chem_kb = db.get(KnowledgeBase, 6)
chem_docs = list(db.scalars(select(Document).where(Document.knowledge_base_id == 6)))
chem_filenames = [d.filename for d in chem_docs]
print("Existing Chem docs:", chem_filenames)

chem_part1_path = Path("C:/Zeus/endsem_proj/NCERT-Chemistry-Class-11-Part 1 ( etc.) (z-library.sk, 1lib.sk, z-lib.sk).pdf")
part1_name = chem_part1_path.name

if part1_name not in chem_filenames and chem_part1_path.exists():
    print(f"Ingesting {part1_name} into Chemistry library...")
    with open(chem_part1_path, "rb") as f:
        content = f.read()
    doc = ingest_pdf(
        db,
        knowledge_base_id=6,
        filename=part1_name,
        content=content,
    )
    print(f"Successfully ingested {part1_name}: {len(doc.chunks)} chunks")

# 5. Summary
kbs = list(db.scalars(select(KnowledgeBase).order_by(KnowledgeBase.name)))
print("\n--- CONTENT LIBRARIES ---")
for k in kbs:
    docs = list(db.scalars(select(Document).where(Document.knowledge_base_id == k.id)))
    print(f"Library [{k.name}] (Subject: {k.subject}, ID: {k.id}):")
    for d in docs:
        print(f"  - {d.filename} ({len(d.chunks)} chunks)")
