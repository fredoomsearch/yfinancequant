from __future__ import annotations

import datetime as dt
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image


OUT_DIR = Path(__file__).resolve().parent
IMG_DIR = OUT_DIR / "images"
DOCX_PATH = OUT_DIR / "evidencia_yfinance.docx"


ENTRIES = [
    {
        "file": "architecture.png",
        "title": "Figura 1. Arquitectura del repositorio",
        "description": "Muestra la organización del proyecto por capas. Se distinguen assistant, pipeline, agents, providers, schemas, scripts, utils y tests.",
        "part": "Corresponde al mapa físico del proyecto y a la explicación de cómo se separa la lógica conversacional, la orquestación y las etapas técnicas.",
    },
    {
        "file": "shortlonghold().png",
        "title": "Figura 2. Señal cuantitativa y lectura de long / short / hold",
        "description": "Resume cómo los modelos locales votan y cómo el orquestador traduce esas votaciones en una decisión final con confianza.",
        "part": "Pertenece a la explicación de la salida del modelado y del significado de la decisión final para negocio.",
    },
    {
        "file": "1.png",
        "title": "Figura 3. Menú principal y contexto actual del asistente",
        "description": "Se ve el banner principal con agentes, modos, atajos y el contexto activo de la corrida. Esta captura demuestra que el chat no es un prompt aislado sino una interfaz con estado.",
        "part": "Corresponde a la capa assistant y a la navegación de alto nivel entre extracción, limpieza, modelado y orquestador.",
    },
    {
        "file": "2.png",
        "title": "Figura 4. Guía de extracción y resumen del proyecto",
        "description": "La captura muestra la explicación de qué símbolo se usa, qué columnas se extraen y cómo el asistente presenta la corrida en lenguaje natural.",
        "part": "Pertenece a la etapa de extracción y a la narrativa de negocio sobre cómo se captura la materia prima del análisis.",
    },
    {
        "file": "3.png",
        "title": "Figura 5. Guía de limpieza y handoff entre etapas",
        "description": "Aquí se explica la limpieza, el propósito de la etapa, las preguntas naturales y la transición desde extracción hacia modelado.",
        "part": "Corresponde a CleaningAgent y al handoff del mismo run hacia el modelo.",
    },
    {
        "file": "4.png",
        "title": "Figura 6. Centro de datos limpios",
        "description": "La vista centraliza símbolos, métricas, filas y esquema. También deja ver la estructura base y la relación entre columnas crudas y columnas listas para el modelo.",
        "part": "Pertenece a la capa clean data / data exploration, útil para revisar el estado del dataset limpio.",
    },
    {
        "file": "5.png",
        "title": "Figura 7. Análisis de una fila limpia de BTC-USD",
        "description": "La captura interpreta una fila concreta de BTC-USD: movimiento de cierre, volumen, volatilidad y lectura local de la señal.",
        "part": "Corresponde a la vista de análisis de fila dentro de los datos limpios.",
    },
    {
        "file": "6.png",
        "title": "Figura 8. Vista de métricas limpias y decisión del run",
        "description": "Se aprecia el resumen del run, la decisión producida y el conjunto de columnas limpias usado para modelar la señal.",
        "part": "Pertenece a la explicación de métricas y a la relación entre limpieza y modelado.",
    },
    {
        "file": "7.png",
        "title": "Figura 9. Guía de modelado",
        "description": "El panel de modelado explica el propósito de la etapa: votos, confianza y señales long/short/hold, junto con el handoff al orquestador.",
        "part": "Corresponde a ModelingAgent y a la presentación ejecutiva de la señal cuantitativa.",
    },
    {
        "file": "8.png",
        "title": "Figura 10. Resultado del modelado y explicación de la señal",
        "description": "Aquí se ve el detalle de las predicciones por modelo, la mayoría, la confianza y la leyenda del significado de long, short y hold.",
        "part": "Pertenece a la capa de decisión cuantitativa y a la explicación técnica del voto del modelo.",
    },
    {
        "file": "9.png",
        "title": "Figura 11. Conexión entre limpieza y decisión",
        "description": "La captura muestra cómo los datos limpios alimentan el modelo y cómo el orquestador usa esa información para decidir.",
        "part": "Corresponde a la historia completa de limpieza -> modelado -> orquestación.",
    },
    {
        "file": "10.png",
        "title": "Figura 12. Evidencia de ajustes finales y puente legacy",
        "description": "Se aprecia material de ajuste final, incluyendo la referencia al puente legacy BTC/ETH/LTC y notas sobre los cambios de documentación.",
        "part": "Pertenece a la parte de compatibilidad y a la evidencia final de entrega.",
    },
    {
        "file": "11.png",
        "title": "Figura 13. Centro de modos y comparación con Binance",
        "description": "La vista documenta los modos disponibles: local_only, compare-binance, --groq-brain y la combinación de ambos.",
        "part": "Corresponde a la capa de modos experimentales y a la explicación de rutas alternativas de ejecución.",
    },
    {
        "file": "12.png",
        "title": "Figura 14. Flujo completo del chat y del run",
        "description": "Esta captura larga reúne menú, extracción, limpieza, modelado y orquestación, mostrando la secuencia completa de la corrida.",
        "part": "Pertenece a la evidencia de end-to-end y al relato de negocio sobre cómo se recorre el sistema.",
    },
    {
        "file": "13.png",
        "title": "Figura 15. Compare-binance y cierre de evidencia",
        "description": "Se observa la corrida con compare-binance y la explicación final de la decisión, útil como cierre de la evidencia visual.",
        "part": "Corresponde al modo experimental con comparación de fuentes y a la trazabilidad de la decisión final.",
    },
]


def w_p(text: str, *, bold: bool = False, size: int | None = None, center: bool = False) -> str:
    rpr = []
    if bold:
        rpr.append("<w:b/>")
    if size is not None:
        rpr.append(f'<w:sz w:val="{size}"/>')
    if rpr:
        rpr_xml = f"<w:rPr>{''.join(rpr)}</w:rPr>"
    else:
        rpr_xml = ""
    ppr = '<w:pPr><w:jc w:val="center"/></w:pPr>' if center else ""
    return (
        "<w:p>"
        f"{ppr}"
        f"<w:r>{rpr_xml}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"
        "</w:p>"
    )


def w_br_page() -> str:
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def image_xml(rel_id: str, file_name: str, px_w: int, px_h: int, idx: int, width_in: float = 6.2) -> str:
    emu_per_inch = 914400
    cx = int(width_in * emu_per_inch)
    cy = int(cx * (px_h / px_w))
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{idx}" name="{escape(file_name)}" descr="{escape(file_name)}"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks noChangeAspect="1"/>
        </wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{escape(file_name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
""".strip()


def build_docx() -> None:
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")

    image_files = [IMG_DIR / entry["file"] for entry in ENTRIES]
    for image_path in image_files:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

    rels = []
    body_parts: list[str] = []

    body_parts.append(w_p("Evidencia visual del proyecto y resumen para respuesta", bold=True, size=34))
    body_parts.append(w_p("Documento preparado para acompañar el zip del proyecto y explicar el flujo, la arquitectura y la narrativa de negocio.", size=20))
    body_parts.append(w_p("Borrador de respuesta al correo", bold=True, size=28))
    body_parts.append(w_p("Buenas tardes, Andrés:"))
    body_parts.append(w_p("Gracias por la atención que le pusieron a la prueba y por el tiempo de entrega anticipado."))
    body_parts.append(w_p("Adjunto una evidencia en Word con las imágenes organizadas y descritas por la parte del proyecto a la que pertenecen, para explicar de forma más clara el flujo, la arquitectura y el valor de negocio."))
    body_parts.append(w_p("Quiero ser transparente: mi enfoque inicial fue demasiado lineal y, en algunos puntos, equivocado en la forma de comunicar el proyecto. Por eso envío esta versión final como cierre, junto con el zip del repositorio, ya ordenado para que se entienda mejor mi trabajo y la intención de la solución."))
    body_parts.append(w_p("Para la reunión, me funciona el jueves en la tarde o el viernes después de las 11:00 a. m."))
    body_parts.append(w_p("Saludos,"))
    body_parts.append(w_p("Julian"))
    body_parts.append(w_br_page())
    body_parts.append(w_p("Nota de contexto", bold=True, size=28))
    body_parts.append(w_p("Mi enfoque inicial fue demasiado lineal y no transmitió con suficiente claridad la arquitectura completa. Esta versión organiza la evidencia por capas y por etapa para que el proyecto se entienda mejor en términos técnicos y de negocio.", size=20))
    body_parts.append(w_br_page())
    body_parts.append(w_p("Capturas y descripciones", bold=True, size=28))

    for idx, entry in enumerate(ENTRIES, start=1):
        image_path = IMG_DIR / entry["file"]
        with Image.open(image_path) as im:
            width_px, height_px = im.size
        rel_id = f"rId{idx}"
        rels.append(f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{entry["file"]}"/>')
        body_parts.append(w_br_page())
        body_parts.append(w_p(entry["title"], bold=True, size=24))
        body_parts.append(image_xml(rel_id, entry["file"], width_px, height_px, idx))
        body_parts.append(w_p(f"Descripción: {entry['description']}", size=20))
        body_parts.append(w_p(f"Parte: {entry['part']}", size=20))

    body_parts.append(w_br_page())
    body_parts.append(w_p("Cómo usar este documento", bold=True, size=28))
    body_parts.append(w_p("• Úsalo como anexo visual para explicar el proyecto en términos de arquitectura y negocio.", size=20))
    body_parts.append(w_p("• Adjúntalo junto con el zip del repositorio como evidencia de cierre.", size=20))
    body_parts.append(w_p("• Si vas a responder el correo, copia el borrador de la primera sección y ajusta la disponibilidad que más te convenga.", size=20))

    document_xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"
    xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"
    xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\"
    xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\"
    xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\">
  <w:body>
    {''.join(body_parts)}
    <w:sectPr>
      <w:pgSz w:w=\"12240\" w:h=\"15840\"/>
      <w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>
      <w:cols w:space=\"708\"/>
      <w:docGrid w:linePitch=\"360\"/>
    </w:sectPr>
  </w:body>
</w:document>
"""

    styles_xml = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">
  <w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\">\n    <w:name w:val=\"Normal\"/>\n    <w:qFormat/>\n  </w:style>
</w:styles>
"""

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        '</Types>',
    ]

    rels_xml = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>
</Relationships>
"""

    document_rels_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        *[f'  {rel}' for rel in rels],
        '</Relationships>',
    ]

    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    core_xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\"
    xmlns:dc=\"http://purl.org/dc/elements/1.1/\"
    xmlns:dcterms=\"http://purl.org/dc/terms/\"
    xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\"
    xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">
  <dc:title>Evidencia visual del proyecto</dc:title>
  <dc:subject>YFinance Quant Assistant</dc:subject>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:created>
  <dcterms:modified xsi:type=\"dcterms:W3CDTF\">{now}</dcterms:modified>
</cp:coreProperties>
"""

    app_xml = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\"
    xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">
  <Application>Codex</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size=\"2\" baseType=\"variant\">
      <vt:variant>
        <vt:lpstr>Title</vt:lpstr>
      </vt:variant>
      <vt:variant>
        <vt:i4>1</vt:i4>
      </vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size=\"1\" baseType=\"lpstr\">
      <vt:lpstr>Evidencia visual del proyecto</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
"""

    with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "\n".join(content_types))
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/_rels/document.xml.rels", "\n".join(document_rels_xml))
        for entry in ENTRIES:
            zf.write(IMG_DIR / entry["file"], arcname=f"word/media/{entry['file']}")

    print(DOCX_PATH)


if __name__ == "__main__":
    build_docx()
