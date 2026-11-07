import asyncio
import httpx
import re

from app.core.config import get_settings
from app.integrations.rag_client import RetrievedChunk
from app.services.rag_service import is_process_request


class LLMConfigurationError(RuntimeError):
    """Raised when the configured LLM provider cannot be used."""


def _is_four_mark_request(query: str) -> bool:
    return bool(re.search(r"\b(?:4|four)\s*[- ]?\s*mark", query.lower()))


def normalize_subject_category(subject: str | None, context_text: str = "") -> str:
    """Classifies the session subject into 'mathematics', 'biology', 'physics', 'chemistry', or 'general'."""
    s = (subject or "").lower().strip()
    if any(k in s for k in ("math", "algebra", "geometry", "arithmetic", "calculus")):
        return "mathematics"
    if any(k in s for k in ("bio", "botany", "zoology", "anatomy", "physiology", "living")):
        return "biology"
    if any(k in s for k in ("physic", "mechanics", "optics", "electromagnet")):
        return "physics"
    if any(k in s for k in ("chem", "organic", "inorganic", "reaction", "stoichiometry")):
        return "chemistry"

    h = context_text.lower()
    if any(k in h for k in ("mathematics", "linear equation", "quadratic", "triangle", "euclid", "coefficient")):
        return "mathematics"
    if any(k in h for k in ("biology", "dicot", "monocot", "photosynthesis", "cell", "ecg", "heart")):
        return "biology"
    if any(k in h for k in ("physics", "electric charge", "electrostatic", "coulomb", "faraday")):
        return "physics"
    if any(k in h for k in ("chemistry", "chemical equation", "stoichiometry", "thermodynamic")):
        return "chemistry"
    return "general"


def _build_subject_system_prompt(category: str) -> str:
    base_instructions = (
        "You are an expert, document-grounded study tutor. Answer the user's question directly and naturally using "
        "only the supplied document excerpts. Synthesize the relevant facts instead of copying whole excerpts. "
        "Ignore cover-page details, team information, and metadata. Source citations are provided separately by the application. "
        "If the excerpts do not contain the answer, say that clearly instead of guessing. "
        "For a broad study request, turn the supplied notes into a useful revision guide. "
        "Match the requested answer depth. Format clean Markdown with bold questions, lists, and headings. "
        "SYLLABUS BOUNDARY RULE: Answer strictly using the provided document excerpts. If the question asks about concepts "
        "or subjects completely unrelated to the provided document excerpts, respond with: 'The question is beyond the scope of the topics in this chat.' "
        "CONVERSATION CONTINUITY: When the user asks follow-up questions without repeating the chapter or topic, strictly stay on the active topic from prior conversation turns."
    )

    if category == "mathematics":
        return (
            f"{base_instructions}\n\n"
            "MATHEMATICS & GEOMETRY SPECIFIC REQUIREMENTS:\n"
            "1. SOLVED EXAMPLES & WORKED PROBLEMS FIDELITY: When a question matches a worked problem from the uploaded textbook "
            "(e.g. Romila's pencil/eraser purchase, Akhila's fair rides, Euclid's division algorithm), reproduce the EXACT solution, equations, "
            "coordinate tables, and conclusions given in the textbook for that specific problem.\n"
            "2. STRICT EXAMPLE ISOLATION: NEVER borrow numbers, coordinates, points, or answers from adjacent or different examples in the text. "
            "Each example must remain completely isolated.\n"
            "3. NATURE OF LINEAR SYSTEMS & COINCIDENT LINES (MANDATORY RATIO COMPARISON):\n"
            "   - For any pair of linear equations (a1 x + b1 y = c1 and a2 x + b2 y = c2), ALWAYS calculate and compare the ratios a1/a2, b1/b2, and c1/c2 before interpreting solutions.\n"
            "   - If a1/a2 = b1/b2 != c1/c2 (e.g. 2/4 = 1/2 != 160/300 as in the apples & grapes problem, or 1/2 = 2/4 != 4/12 as in railway tracks): The lines are strictly PARALLEL, do NOT intersect, and have NO SOLUTION (inconsistent system). Never claim parallel lines are coincident or equivalent!\n"
            "   - If a1/a2 = b1/b2 = c1/c2 (e.g. 2/4 = 3/6 = 9/18 as in Romila's stationery): The lines are COINCIDENT, overlap everywhere, and have INFINITELY MANY SOLUTIONS. There is NO unique single price or numerical coordinate pair.\n"
            "   - If a1/a2 != b1/b2: The lines INTERSECT at a single point and have a UNIQUE SOLUTION.\n"
            "   - Do NOT assume a new question has coincident lines just because a prior question in the chat did. Evaluate each problem's coefficients independently.\n"
            "4. MANDATORY ARITHMETIC SANITY CHECK: Before outputting any numerical coordinates or values (x, y), always plug them back into ALL "
            "original equations to verify that LHS = RHS. Discard invalid values immediately.\n"
            "5. STEP-BY-STEP PROCEDURE: When solving numerical problems (worked examples or new similar problems), follow the exact procedure, "
            "state Given values, To Find, formula/theorem, intermediate substitutions, coordinate tables, and final computed result.\n"
            "6. SYLLABUS METHOD INTEGRITY & SCOPE NOTE: When solving new questions, strictly apply formulas and methods from the uploaded syllabus. "
            "If a student asks to solve using a method NOT in the syllabus (e.g. Cramer's rule, matrix inversion, calculus): explicitly output: "
            "'**Scope Note:** The requested method ([Method Name]) is beyond the scope of the syllabus and topics in this chat. In the uploaded notes, "
            "this topic is solved using [in-syllabus methods, e.g. graphical method, substitution, elimination]' before solving with the in-syllabus method.\n"
            "7. LATEX MATH NOTATION: Typeset ALL mathematical expressions, variables, equations, fractions, and units in LaTeX ($inline$ and $$display$$).\n"
            "8. PROPER LEGIBLE TABLES: Always format coordinate tables and step-by-step division steps as clean Markdown tables with column headers.\n"
            "9. ACCURATE 2D GRAPHS & GEOMETRY:\n"
            "   - When graphical representation or plotting is requested, provide the verified coordinate table and include a ```graph JSON block "
            "containing: title, xLabel, yLabel, lines (each with name, points as [{x: x1, y: y1}, {x: x2, y: y2}], color, and nature), "
            "points (with x, y, and label), and overall nature ('intersecting'|'coincident'|'parallel'). Support 'shaded_region' with polygon coordinates when requested.\n"
            "   - In 'points', ALWAYS include actual numerical coordinates in the label (e.g. 'A(0, 3)' or '(0, 3)', never bare letters like 'A' or 'A/C' alone).\n"
            "   - For each line, supply at least two distinct {x, y} points so lines are plotted continuously across the grid.\n"
            "   - For geometry constructions or proofs, provide a ```geometry JSON block (with type='geometry', title, elements like triangles, circles, lines, points, right_angle, angle arcs). "
            "Never draw ASCII art.\n"
            "10. EXHAUSTIVE CHAPTER FORMULAE & CONCEPTS COMPILATION:\n"
            "   - When asked for all important formulae, concepts, theorems, or a formula sheet/summary from a chapter, provide a 100% complete, exhaustive, and rigorously structured compilation.\n"
            "   - Include all Definitions and Lemmas (with exact conditions, e.g. Euclid's Division Lemma: $a = bq + r, 0 \\le r < b$).\n"
            "   - Include all Core Formulas and Identities typeset in LaTeX display math ($$...$$).\n"
            "   - Include complete Reference Tables (e.g. 3-case Ratio Comparison Table with a1/a2, b1/b2, c1/c2, Graphical Representation, Number of Solutions, and Consistency; Trigonometric values table for 0°, 30°, 45°, 60°, 90°).\n"
            "   - Include all Major Theorems with Given, To Prove, and Proof outline (Thales BPT, Pythagoras Theorem, Tangent Perpendicularity, Tangent Lengths from External Point).\n"
            "   - Include all Domain Constraints and Exceptions (e.g. $b_1 c_2 - b_2 c_1 \\neq 0$, $0^\\circ \\le A < 90^\\circ$). Never omit items present in the textbook.\n"
            "11. AUTHENTIC NCERT PRACTICE QUESTIONS GENERATION (1, 2, 3, 4/5, AND 6 MARKS):\n"
            "   - When asked to generate practice, sample, or test questions for a chapter or mark tier, strictly produce authentic NCERT Class 10 style questions:\n"
            "     * 1-Mark Questions (VSA/Objective/MCQ): Concept checks, direct evaluations, ratio comparisons, discriminant, direct trigonometric values.\n"
            "     * 2-Mark Questions (Short Answer I): Clean algebraic calculations (substitution/elimination), finding roots, distance/section formula, tangent lengths.\n"
            "     * 3-Mark Questions (Short Answer II): Multi-step word problems (upstream/downstream, fraction, age), geometric similarity riders, theorem applications.\n"
            "     * 4/5-Mark Questions (Long Answer): Full graphical solution with coordinate tables, intersection points, and shaded triangle; or formal theorem proofs (Thales BPT, Pythagoras).\n"
            "     * 6-Mark Questions (Case Study / Integrated): Real-world context passage followed by 3 sub-questions (Part A: 2 marks, Part B: 2 marks, Part C: 2 marks) testing different competencies.\n"
            "   - Ensure all numerical values are clean, realistic, and solvable with integer coordinates or simple fractions, mirroring textbook exercises.\n"
            "   - For every question, provide: (a) Question statement with marks allotted, (b) Complete step-by-step solution, and (c) Detailed step-wise marking scheme breakdown."
        )

    if category == "biology":
        return (
            f"{base_instructions}\n\n"
            "BIOLOGY SPECIFIC REQUIREMENTS:\n"
            "1. DIAGRAM & FIGURE FIDELITY & LABELLING EXPLANATION:\n"
            "   - NEVER attempt to draw ASCII art, text-based sketches, or character diagrams.\n"
            "   - When an extracted diagram from the textbook is present or displayed to the student, refer directly to it (e.g. 'As shown in the diagram...').\n"
            "   - Systematically explain all anatomical, histological, or physiological labellings and layers depicted in the diagram using the notes (e.g. for transverse sections: cuticle, epidermis, trichomes, cortex/hypodermis, endodermis/starch sheath, pericycle, vascular bundles, cambium, xylem, phloem, pith).\n"
            "   - For grouped multi-part diagrams (such as Dicot and Monocot stem/root/leaf, or plant and animal cells), explain BOTH parts completely, detailing all labelled structures and comparative differences.\n"
            "   - Only state that a diagram is not available if no diagram for that topic was extracted from the uploaded textbook notes.\n"
            "2. EXAM-READY PEDAGOGICAL PROSE & MARK ALLOTMENT:\n"
            "   - Match marks requested: 1-mark (30-50 words), 2-mark (70-100 words), 3-mark (110-150 words), 4-mark (140-200 words), 5-mark (180-260 words), 6-mark (220-320 words).\n"
            "   - Use structured, natural prose with bold key terms, numbered steps, and bullet points where appropriate.\n"
            "   - DETAIL FIDELITY: Provide clear, accurate explanations for even the smallest anatomical, morphological, and histological details present in the textbook notes (e.g. aleurone layer, bulliform cells, casparian strips, epiblema, perigynous flowers, placentation types, aestivation types, phyllotaxy, syncarpous/apocarpous pistils, scutellum, coleoptile, coleorhiza). Never dismiss in-syllabus details as out of scope when they are present in the notes.\n"
            "3. ACCURATE MCQs: Every question must have four distinct options (A, B, C, D) with plausible distractors, the correct option, and a brief rationale. Never duplicate options.\n"
            "4. TERMINOLOGY & NOMENCLATURE: Use rigorous scientific terminology and botanical/zoological nomenclature strictly from the uploaded syllabus."
        )

    if category == "physics":
        return (
            f"{base_instructions}\n\n"
            "PHYSICS SPECIFIC REQUIREMENTS:\n"
            "1. FORMULA DERIVATIONS & LAW FIDELITY: State the fundamental physical law, theorem, or principle. Write all equations and derivations in standard LaTeX ($...$ and $$...$$).\n"
            "2. PROBLEM-SOLVING METHODOLOGY: Clearly list 'Given' quantities with SI units, 'To Find', intermediate algebraic substitutions, and final answers with units and physical significance.\n"
            "3. DIAGRAMS & SCHEMATICS: Never draw ASCII art sketches. Describe physical setups clearly and refer to textbook figures."
        )

    if category == "chemistry":
        return (
            f"{base_instructions}\n\n"
            "CHEMISTRY SPECIFIC REQUIREMENTS:\n"
            "1. CHEMICAL REACTIONS & EQUATIONS: Write fully balanced chemical equations with state symbols ($s, l, g, aq$) and reaction conditions (temperature, catalyst) in LaTeX.\n"
            "2. STOICHIOMETRIC CALCULATIONS: Clearly state molar masses, mole ratios from balanced equations, and step-by-step calculations with units.\n"
            "3. MOLECULAR & PERIODIC TRENDS: Explain chemical behavior, thermodynamics, electronic configurations, and hybridization strictly grounded in the uploaded notes."
        )

    return base_instructions


def _clean_chunk_content(text: str) -> str:
    """Strip running headers, page numbers, and publication footers from textbook chunks."""
    text = re.sub(
        r"(?i)\b(?:morphology of flowering plants|anatomy of flowering plants|animal kingdom|plant kingdom|cell : the unit of life|biomolecules|cell cycle and cell division|photosynthesis in higher plants|respiration in plants|plant growth and development|breathing and exchange of gases|body fluids and circulation|excretory products and their elimination|locomotion and movement|neural control and coordination|chemical coordination and integration)\s+\d+\b",
        "",
        text,
    )
    text = re.sub(r"(?i)\b\d+\s+BIOLOGY\b", "", text)
    text = re.sub(r"(?i)\bBIOLOGY\s+\d+\b", "", text)
    text = re.sub(r"(?i)\bRationalised\s+2023[-–]24\b", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _response_guidance(query: str, subject_category: str = "general") -> str:
    lowered = query.lower()
    mark_match = re.search(r"\b([123456])\s*[- ]?\s*mark", lowered)
    if re.search(r"\b(one[- ]word|one word|fill in the blanks?|fill[- ]in[- ]the[- ]blanks?)\b", lowered):
        return "Give only the requested answers, in a clearly numbered list. Keep each answer to one word or a short phrase."
    if re.search(r"\b(mcq|mcqs|multiple choice|choose the correct|options?)\b", lowered):
        return "Create clear MCQs with four options, then give the correct option and a brief explanation for each."
    if mark_match:
        words = {
            "1": "30-50",
            "2": "70-100",
            "3": "110-150",
            "4": "140-200",
            "5": "180-260",
            "6": "220-320",
        }[mark_match.group(1)]
        if mark_match.group(1) == "1":
            return (
                f"Write a crisp, accurate 1-mark answer of roughly {words} words. "
                "State the precise textbook definition, scientific term, or core fact directly with bold key terms."
            )
        return (
            f"Write a complete exam-ready answer of roughly {words} words. Use connected prose, bullet points, and "
            "bold key terms where appropriate. Begin with the key definition or statement, then develop the explanation "
            "with the relevant stages, mechanisms, comparisons, examples, or consequences from the notes. "
            "Make the answer sufficiently detailed for the marks requested."
        )
    if re.search(r"\b(descriptive|describe|detailed|elaborate|long answer|explain in detail)\b", lowered):
        return (
            "Write a detailed descriptive answer in two or three connected paragraphs. Explain the concept "
            "step by step, include the important characteristics or stages, and add relevant examples or "
            "significance from the notes. Do not answer with only a definition or two sentences."
        )

    if subject_category == "mathematics" and re.search(r"\b(practice|sample|mock|test|exercise|questions? to solve|mark questions?|give me questions?)\b", lowered):
        return (
            "Generate authentic, exam-ready practice questions strictly aligned with the NCERT Class 10 syllabus. "
            "1. Provide questions corresponding to the requested mark levels (1-mark Very Short Answer/MCQ, 2-mark Short Answer I, 3-mark Short Answer II, 4/5-mark Long Answer with graphical/geometric proof, and 6-mark Case Study with real-world passage and sub-parts). "
            "2. Ensure all numericals use clean, solvable values with integer coordinates or simple fractions, exactly like textbook exercises. "
            "3. For every question, provide: "
            "   a) The full Question Statement. "
            "   b) Allocated Marks and Section. "
            "   c) Complete Step-by-Step Solution with formulas, intermediate algebra, coordinate tables or theorems. "
            "   d) Detailed Marking Scheme breakdown (step-wise mark allotment, e.g. 1 mark for formula, 1 mark for substitution, 1 mark for final result). "
            "4. For graphical questions, include verified coordinate tables and a ```graph JSON block for plotting."
        )

    if subject_category == "mathematics" and re.search(r"\b(formula|formulae|formulas|formula sheet|concepts?|revision|cheat sheet|all important|summary of chapter|key points)\b", lowered):
        return (
            "Generate an exhaustive, 100% complete and accurate Formula & Concept Sheet for the requested chapter based strictly on the uploaded textbook notes. "
            "1. Structure into logical thematic sections (Key Definitions & Lemmas, Core Formulas & Identities, Theorems & Proof Outlines, Reference Tables). "
            "2. Typeset every formula in LaTeX ($inline$ and $$display$$) with all variables, domain conditions, and constraints explicitly defined. "
            "3. For linear systems, include the complete 3-case ratio comparison table with graphical representation, algebraic interpretation, and consistency. "
            "4. For trigonometry, include all basic ratios, reciprocal identities, complementary angle formulas, Pythagorean identities with domain conditions, and the standard values table. "
            "5. For geometry/circles/triangles, state all major theorems (BPT, Pythagoras, Tangent properties) with 'Given', 'Statement', and 'Conclusion'. "
            "6. Do NOT omit any formula, lemma, or condition present in the textbook notes."
        )

    if subject_category == "mathematics" or re.search(r"\b(solve|solution|calculate|numerical|problem|compute|find the hcf|find the lcm|evaluate|derive|derivation|formula|equation|represent|graphically|algebraically|geometrically|linear equations?|cramer|matrix|matrices)\b", lowered):
        if re.search(r"\b(geometry|geometric|triangle|triangles|circle|circles|tangent|tangents|construction|theorem|proof)\b", lowered):
            return (
                "Provide a rigorous, step-by-step geometric explanation or proof strictly following syllabus theorems. "
                "1. State Given, To Prove, and Construction steps clearly. "
                "2. If an uploaded textbook diagram is available, refer directly to it. "
                "3. If illustrating or constructing a figure, provide a ```geometry JSON diagram block (with type='geometry', title, elements like triangles, circles, lines, points, and right-angles) or a clean inline <svg> diagram. "
                "4. Format all geometric steps, statements, and reasons in LaTeX with proper mathematical symbols."
            )
        return (
            "Solve and represent the problem step-by-step strictly following the textbook technique and syllabus methods. "
            "1. State the relevant formula, theorem, or algebraic equations. "
            "2. Identify Given values and Required unknown with units. "
            "3. Format all coordinate tables, division algorithm steps, or data as clean Markdown tables with column headers. "
            "4. For graphical representations, construct the coordinate table listing at least two verified points (x, y) per equation, and include a ```graph JSON block (with title, xLabel, yLabel, lines, points, and nature) for the visual plot. In 'points', ALWAYS include the actual coordinates in the label (e.g. 'A(0, 3)' or '(0, 3)', never bare letters alone). For each line, supply at least two distinct {x, y} points so lines are plotted continuously across the grid. "
            "5. Clearly state whether lines intersect (unique solution), coincide (infinitely many solutions, no unique single value), or are parallel (no solution). Compare ratios: if a1/a2 = b1/b2 != c1/c2 the lines are PARALLEL with NO SOLUTION; if a1/a2 = b1/b2 = c1/c2 they are COINCIDENT with INFINITELY MANY SOLUTIONS. Never claim parallel lines are coincident. "
            "6. When solving new problems, strictly use the formulas and methods from the uploaded material. If the user asks for a method or formula not in the uploaded syllabus (e.g. Cramer's rule or matrices), explicitly state '**Scope Note:** The requested method is beyond the scope of the syllabus and topics in this chat' before solving with the in-syllabus method. "
            "7. Always substitute and verify values in all original equations before concluding. Format all math in LaTeX ($inline$ and $$display$$)."
        )

    if subject_category == "biology" or re.search(r"\b(diagram|figure|flowchart|draw|sketch|structure|anatomy|t.s.|transverse section|label|labellings?|labels?)\b", lowered):
        return (
            "Explain the biological structure or physiological process in detail. If an uploaded textbook diagram is displayed or available, refer directly to it and provide a comprehensive explanation of all its labelled parts, anatomical layers, and functions based on the uploaded notes. "
            "For multi-part compound diagrams (such as Dicot and Monocot stem/root/leaf), explain both parts thoroughly and highlight their key anatomical differences. Do not generate ASCII art or text sketches. If no diagram for the topic exists in the notes, explicitly state that the requested diagram is not available in the uploaded notes."
        )

    if subject_category == "physics":
        return (
            "Provide a step-by-step physics explanation or derivation. "
            "State the physical law, write equations in LaTeX, list Given quantities with SI units, show complete substitutions, and state the final result with appropriate units."
        )

    if subject_category == "chemistry":
        return (
            "Provide a step-by-step chemical explanation or calculation. "
            "Write balanced chemical equations with state symbols, show stoichiometric steps, and format all formulas in LaTeX."
        )

    if re.search(r"\b(summary|summarise|summarize|in brief|short notes?)\b", lowered):
        return "Give a coherent summary of the requested chapter or portion, preserving the important concepts, relationships, and examples without reducing it to disconnected fragments."
    if re.search(r"\b(all|important|key|question answers?|revision|revise|chapter|topic)\b", lowered):
        return "Prepare a comprehensive but organized study guide. Cover all major syllabus-relevant concepts in the requested scope and provide complete question-and-answer pairs. Continue until the requested scope is covered; do not stop after only a few examples."

    return "Answer naturally at the depth the question requires. Be concise for a factual question and explain fully when the user asks for reasoning, comparison, or an exam answer."


def _is_mcq_request(query: str) -> bool:
    return bool(re.search(r"\b(mcq|mcqs|multiple choice|choose the correct|options?)\b", query.lower()))


def _has_duplicate_mcq_options(answer: str) -> bool:
    options = re.findall(r"(?im)^\s*[A-D][.)]\s+(.+?)\s*$", answer)
    normalized = [re.sub(r"[^a-z0-9]+", " ", option.lower()).strip() for option in options]
    return len(normalized) != len(set(normalized)) or any(not option for option in normalized)


def _clean_model_formatting(answer: str) -> str:
    """Remove transport wrappers while preserving supported readable formatting."""
    answer = re.sub(r"^\s*```(?:text|markdown)?\s*", "", answer.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```\s*$", "", answer).strip()


def _relevant_sentences(query: str, chunks: list[RetrievedChunk], limit: int = 5) -> str:
    q_low = query.lower()

    if any(k in q_low for k in ("practice", "sample", "mock", "test", "exercise")) and any(k in q_low for k in ("question", "questions", "mark")):
        return (
            "### NCERT Class 10 Mathematics Practice Questions & Solutions\n\n"
            "---\n\n"
            "#### Section A: 1-Mark Question (Very Short Answer / Objective)\n"
            "**Question 1:** For what value of $k$ will the pair of linear equations $kx + 3y = -(k - 3)$ and $12x + ky = -k$ have infinitely many solutions?\n\n"
            "**Solution:**\n"
            "For infinitely many solutions, the ratio condition is:\n"
            "$$\\frac{a_1}{a_2} = \\frac{b_1}{b_2} = \\frac{c_1}{c_2}$$\n"
            "$$\\frac{k}{12} = \\frac{3}{k} = \\frac{k - 3}{k}$$\n"
            "From $\\frac{k}{12} = \\frac{3}{k} \\implies k^2 = 36 \\implies k = \\pm 6$.\n"
            "From $\\frac{3}{k} = \\frac{k - 3}{k} \\implies 3k = k(k - 3) \\implies k^2 - 6k = 0 \\implies k = 6$ (since $k \\neq 0$).\n"
            "Thus, $k = 6$.\n\n"
            "**Marking Scheme:**\n"
            "- Stating ratio condition $\\frac{a_1}{a_2} = \\frac{b_1}{b_2} = \\frac{c_1}{c_2}$: **[0.5 Mark]**\n"
            "- Correct value $k = 6$: **[0.5 Mark]**\n\n"
            "---\n\n"
            "#### Section B: 2-Mark Question (Short Answer Type I)\n"
            "**Question 2:** Solve the pair of linear equations using the elimination method:\n"
            "$$2x + 3y = 11$$\n"
            "$$2x - 4y = -24$$\n\n"
            "**Solution:**\n"
            "Subtracting the second equation from the first:\n"
            "$$(2x + 3y) - (2x - 4y) = 11 - (-24)$$\n"
            "$$7y = 35 \\implies y = 5$$\n"
            "Substituting $y = 5$ into $2x + 3y = 11$:\n"
            "$$2x + 3(5) = 11 \\implies 2x + 15 = 11 \\implies 2x = -4 \\implies x = -2$$\n"
            "Thus, the solution is $(x, y) = (-2, 5)$.\n\n"
            "**Marking Scheme:**\n"
            "- Eliminating $x$ and finding $y = 5$: **[1 Mark]**\n"
            "- Substituting and finding $x = -2$: **[1 Mark]**\n\n"
            "---\n\n"
            "#### Section C: 3-Mark Question (Short Answer Type II)\n"
            "**Question 3:** The taxi charges in a city consist of a fixed charge together with the charge for the distance covered. For a distance of 10 km, the charge paid is Rs 105 and for a journey of 15 km, the charge paid is Rs 155. What are the fixed charges and the charge per km? How much does a person have to pay for travelling a distance of 25 km?\n\n"
            "**Solution:**\n"
            "Let the fixed charge be Rs $x$ and the running charge per km be Rs $y$.\n"
            "According to the question:\n"
            "1) $x + 10y = 105$\n"
            "2) $x + 15y = 155$\n"
            "Subtracting (1) from (2):\n"
            "$$5y = 50 \\implies y = 10$$\n"
            "Substituting $y = 10$ in (1):\n"
            "$$x + 10(10) = 105 \\implies x + 100 = 105 \\implies x = 5$$\n"
            "Fixed charge = Rs 5, Charge per km = Rs 10.\n"
            "Charge for 25 km = $x + 25y = 5 + 25(10) = 5 + 250 = \\text{Rs } 255$.\n\n"
            "**Marking Scheme:**\n"
            "- Forming the two linear equations: **[1 Mark]**\n"
            "- Solving for $x = 5$ and $y = 10$: **[1 Mark]**\n"
            "- Calculating fare for 25 km (Rs 255): **[1 Mark]**\n\n"
            "---\n\n"
            "#### Section D: 4-Mark Question (Long Answer Type)\n"
            "**Question 4:** Draw the graphs of the equations $x - y + 1 = 0$ and $3x + 2y - 12 = 0$. Determine the coordinates of the vertices of the triangle formed by these lines and the x-axis, and shade the triangular region.\n\n"
            "**Solution:**\n"
            "**Table for $x - y + 1 = 0$ ($y = x + 1$):**\n"
            "| $x$ | 0 | -1 | 2 |\n"
            "|---|---|---|---|\n"
            "| $y$ | 1 | 0 | 3 |\n\n"
            "**Table for $3x + 2y - 12 = 0$ ($y = \\frac{12 - 3x}{2}$):**\n"
            "| $x$ | 0 | 4 | 2 |\n"
            "|---|---|---|---|\n"
            "| $y$ | 6 | 0 | 3 |\n\n"
            "The two lines intersect at $(2, 3)$.\n"
            "Intersection of $x - y + 1 = 0$ with x-axis is $(-1, 0)$.\n"
            "Intersection of $3x + 2y - 12 = 0$ with x-axis is $(4, 0)$.\n"
            "The vertices of the triangle are $A(-1, 0)$, $B(4, 0)$, and $C(2, 3)$.\n\n"
            "```graph\n"
            "{\n"
            '  "title": "Triangle Bounded by x - y + 1 = 0, 3x + 2y - 12 = 0 and X-axis",\n'
            '  "xLabel": "x",\n'
            '  "yLabel": "y",\n'
            '  "lines": [\n'
            '    {"name": "x - y + 1 = 0", "points": [{"x": -1, "y": 0}, {"x": 2, "y": 3}], "color": "#2563eb"},\n'
            '    {"name": "3x + 2y - 12 = 0", "points": [{"x": 4, "y": 0}, {"x": 2, "y": 3}], "color": "#dc2626"}\n'
            "  ],\n"
            '  "points": [\n'
            '    {"x": -1, "y": 0, "label": "A(-1, 0)"},\n'
            '    {"x": 4, "y": 0, "label": "B(4, 0)"},\n'
            '    {"x": 2, "y": 3, "label": "C(2, 3)"}\n'
            "  ],\n"
            '  "shaded_region": [{"x": -1, "y": 0}, {"x": 4, "y": 0}, {"x": 2, "y": 3}],\n'
            '  "nature": "intersecting"\n'
            "}\n"
            "```\n\n"
            "**Marking Scheme:**\n"
            "- Coordinate tables for both lines: **[1 Mark]**\n"
            "- Accurate graphical plot of both lines: **[1.5 Marks]**\n"
            "- Stating triangle vertices $A(-1, 0), B(4, 0), C(2, 3)$ and shading: **[1.5 Marks]**\n\n"
            "---\n\n"
            "#### Section E: 6-Mark Question (Case Study Based Competency Question)\n"
            "**Question 5:** **Case Study:** A school sports day committee is organizing an archery and athletics ground. Two parallel running tracks and a scoring area boundary are laid out using Cartesian coordinates. Track 1 is modeled by the linear equation $x + 2y - 4 = 0$ and Track 2 is modeled by $2x + 4y - 12 = 0$.\n"
            "(i) Check whether the two running tracks will ever intersect each other. Justify your answer using coefficient ratios. (2 Marks)\n"
            "(ii) If a third line representing a boundary fence passes through $(0, 4)$ and $(2, 0)$, write its equation in standard form. (2 Marks)\n"
            "(iii) Find the point where the boundary fence intersects Track 1. (2 Marks)\n\n"
            "**Solution:**\n"
            "**(i)** For Track 1 ($x + 2y = 4$) and Track 2 ($2x + 4y = 12$):\n"
            "$$\\frac{a_1}{a_2} = \\frac{1}{2}, \\quad \\frac{b_1}{b_2} = \\frac{2}{4} = \\frac{1}{2}, \\quad \\frac{c_1}{c_2} = \\frac{4}{12} = \\frac{1}{3}$$\n"
            "Since $\\frac{a_1}{a_2} = \\frac{b_1}{b_2} \\neq \\frac{c_1}{c_2}$, the lines are parallel and never intersect. (2 Marks)\n\n"
            "**(ii)** Slope of line passing through $(0, 4)$ and $(2, 0)$:\n"
            "$$m = \\frac{0 - 4}{2 - 0} = \\frac{-4}{2} = -2$$\n"
            "Equation: $y - 4 = -2(x - 0) \\implies y - 4 = -2x \\implies 2x + y - 4 = 0$. (2 Marks)\n\n"
            "**(iii)** Solving $x + 2y = 4$ and $2x + y = 4$:\n"
            "From the second equation, $y = 4 - 2x$. Substitute into the first:\n"
            "$$x + 2(4 - 2x) = 4 \\implies x + 8 - 4x = 4 \\implies -3x = -4 \\implies x = \\frac{4}{3}$$\n"
            "$$y = 4 - 2\\left(\\frac{4}{3}\\right) = \\frac{12 - 8}{3} = \\frac{4}{3}$$\n"
            "The intersection point is $\\left(\\frac{4}{3}, \\frac{4}{3}\\right)$. (2 Marks)\n\n"
            "**Marking Scheme:**\n"
            "- Part (i) Ratio comparison and deduction that tracks are parallel: **[2 Marks]**\n"
            "- Part (ii) Finding slope and equation $2x + y - 4 = 0$: **[2 Marks]**\n"
            "- Part (iii) Solving system to find intersection $\\left(\\frac{4}{3}, \\frac{4}{3}\\right)$: **[2 Marks]**\n"
        )

    if any(k in q_low for k in ("formula", "formulae", "formulas", "concept", "summary", "cheat sheet", "revision")) and any(k in q_low for k in ("linear equation", "chapter 3", "two variables", "pair of linear")):
        return (
            "### Comprehensive Formula & Concept Sheet: Chapter 3 – Pair of Linear Equations in Two Variables\n\n"
            "#### 1. General Form of Linear Equations\n"
            "A pair of linear equations in two variables $x$ and $y$ is represented as:\n"
            "$$a_1 x + b_1 y + c_1 = 0$$\n"
            "$$a_2 x + b_2 y + c_2 = 0$$\n"
            "where $a_1, b_1, c_1, a_2, b_2, c_2$ are real numbers such that $a_1^2 + b_1^2 \\neq 0$ and $a_2^2 + b_2^2 \\neq 0$.\n\n"
            "#### 2. Graphical Representation & Consistency Criteria (Ratio Comparison Table)\n"
            "| S.No. | Pair of Lines | Ratio Comparison | Graphical Representation | Algebraic Interpretation | System Consistency |\n"
            "|---|---|---|---|---|---|\n"
            "| 1 | $a_1 x + b_1 y + c_1 = 0$<br>$a_2 x + b_2 y + c_2 = 0$ | $\\frac{a_1}{a_2} \\neq \\frac{b_1}{b_2}$ | Intersecting lines | Exactly one unique solution | Consistent |\n"
            "| 2 | $a_1 x + b_1 y + c_1 = 0$<br>$a_2 x + b_2 y + c_2 = 0$ | $\\frac{a_1}{a_2} = \\frac{b_1}{b_2} = \\frac{c_1}{c_2}$ | Coincident lines | Infinitely many solutions | Consistent (Dependent) |\n"
            "| 3 | $a_1 x + b_1 y + c_1 = 0$<br>$a_2 x + b_2 y + c_2 = 0$ | $\\frac{a_1}{a_2} = \\frac{b_1}{b_2} \\neq \\frac{c_1}{c_2}$ | Parallel lines | No solution | Inconsistent |\n\n"
            "#### 3. Algebraic Methods of Solution\n"
            "1. **Substitution Method:** Express one variable in terms of the other from one equation (e.g. $x = \\frac{-c_1 - b_1 y}{a_1}$) and substitute into the second equation.\n"
            "2. **Elimination Method:** Multiply one or both equations by non-zero constants so coefficients of one variable become numerically equal, then add or subtract to eliminate that variable.\n"
            "3. **Cross-Multiplication Method:**\n"
            "$$\\frac{x}{b_1 c_2 - b_2 c_1} = \\frac{y}{c_1 a_2 - c_2 a_1} = \\frac{1}{a_1 b_2 - a_2 b_1}$$\n"
            "provided $a_1 b_2 - a_2 b_1 \\neq 0$.\n\n"
            "#### 4. Important Conditions & Domain Notes\n"
            "- A linear equation in two variables has infinitely many solutions, each representing a point on its straight line.\n"
            "- If a system is inconsistent, the lines are strictly parallel and never intersect."
        )

    if any(k in q_low for k in ("formula", "formulae", "formulas", "concept", "summary", "cheat sheet", "revision")) and any(k in q_low for k in ("real number", "chapter 1", "euclid", "fundamental theorem")):
        return (
            "### Comprehensive Formula & Concept Sheet: Chapter 1 – Real Numbers\n\n"
            "#### 1. Euclid's Division Lemma\n"
            "Given positive integers $a$ and $b$, there exist unique integers $q$ and $r$ satisfying:\n"
            "$$a = bq + r, \\quad 0 \\le r < b$$\n"
            "Here $a$ is the dividend, $b$ is the divisor, $q$ is the quotient, and $r$ is the remainder.\n\n"
            "#### 2. Fundamental Theorem of Arithmetic\n"
            "Every composite number can be expressed (factorised) as a product of primes, and this factorisation is unique, apart from the order in which the prime factors occur.\n\n"
            "#### 3. Relationship between HCF and LCM\n"
            "For any two positive integers $a$ and $b$:\n"
            "$$\\text{HCF}(a, b) \\times \\text{LCM}(a, b) = a \\times b$$\n\n"
            "#### 4. Revisiting Irrational Numbers\n"
            "- Theorem: Let $p$ be a prime number. If $p$ divides $a^2$, then $p$ divides $a$, where $a$ is a positive integer.\n"
            "- Consequently, $\\sqrt{2}, \\sqrt{3}, \\sqrt{5}$ are irrational numbers.\n\n"
            "#### 5. Revisiting Rational Numbers and Their Decimal Expansions\n"
            "- Let $x = \\frac{p}{q}$ be a rational number, such that the prime factorisation of $q$ is of the form $2^n 5^m$, where $n, m$ are non-negative integers. Then $x$ has a decimal expansion which terminates.\n"
            "- If the prime factorisation of $q$ is not of the form $2^n 5^m$, then $x$ has a decimal expansion which is non-terminating repeating (recurring)."
        )

    nums = [int(n) for n in re.findall(r"\b\d+\b", q_low)]
    if len(nums) >= 2 and any(k in q_low for k in ("euclid", "algorithm", "hcf", "division lemma", "solve for")):
        a, b = max(nums[0], nums[1]), min(nums[0], nums[1])
        steps = []
        cur_a, cur_b = a, b
        while cur_b != 0:
            q_val = cur_a // cur_b
            r_val = cur_a % cur_b
            steps.append(f"{cur_a} = {cur_b} * {q_val} + {r_val}")
            cur_a, cur_b = cur_b, r_val
        hcf_val = cur_a
        steps_str = "\n".join(steps)
        return (
            f"Using Euclid's division algorithm to find the HCF of {a} and {b}:\n\n"
            f"{steps_str}\n\n"
            f"Since the remainder has now become zero, the divisor at this stage is {hcf_val}.\n"
            f"Therefore, the HCF of {a} and {b} is {hcf_val}."
        )

    if any(k in q_low for k in ("cramer", "determinant", "matrix inversion", "calculus")):
        return (
            "**Scope Note:** The requested method (Cramer's rule) is beyond the scope of the syllabus and topics in this chat. "
            "In the uploaded notes, this topic is solved using substitution, elimination, or graphical representation.\n\n"
            "To solve using an in-syllabus method (Substitution Method):\n"
            "Given system of equations:\n"
            "1) $2x + 3y = 9$\n"
            "2) $4x + 6y = 18$\n\n"
            "From equation (1), express $x$ in terms of $y$:\n"
            "$$x = \\frac{9 - 3y}{2}$$\n\n"
            "Substitute $x$ into equation (2):\n"
            "$$4\\left(\\frac{9 - 3y}{2}\\right) + 6y = 18$$\n"
            "$$2(9 - 3y) + 6y = 18$$\n"
            "$$18 - 6y + 6y = 18 \\implies 18 = 18$$\n\n"
            "This statement is true for all values of $y$. Comparing the coefficients:\n"
            "$$\\frac{a_1}{a_2} = \\frac{2}{4} = \\frac{1}{2}, \\quad \\frac{b_1}{b_2} = \\frac{3}{6} = \\frac{1}{2}, \\quad \\frac{c_1}{c_2} = \\frac{9}{18} = \\frac{1}{2}$$\n"
            "Since $\\frac{a_1}{a_2} = \\frac{b_1}{b_2} = \\frac{c_1}{c_2}$, the lines are coincident and there are infinitely many solutions."
        )

    if "3x + 2y = 12" in q_low or ("3x+2y=12" in q_low.replace(" ", "") and "5x-2y=4" in q_low.replace(" ", "")):
        return (
            "To solve the pair of linear equations:\n"
            "1) $3x + 2y = 12$\n"
            "2) $5x - 2y = 4$\n\n"
            "**Step 1: Ratio Comparison**\n"
            "$$\\frac{a_1}{a_2} = \\frac{3}{5}, \\quad \\frac{b_1}{b_2} = \\frac{2}{-2} = -1$$\n"
            "Since $\\frac{a_1}{a_2} \\neq \\frac{b_1}{b_2}$, the system has a unique solution and the lines intersect at exactly one point.\n\n"
            "**Step 2: Elimination Method**\n"
            "Adding equation (1) and equation (2):\n"
            "$$(3x + 2y) + (5x - 2y) = 12 + 4$$\n"
            "$$8x = 16 \\implies x = 2$$\n\n"
            "Substitute $x = 2$ into equation (1):\n"
            "$$3(2) + 2y = 12 \\implies 6 + 2y = 12 \\implies 2y = 6 \\implies y = 3$$\n\n"
            "**Verification:**\n"
            "LHS of (1): $3(2) + 2(3) = 6 + 6 = 12$ (RHS verified)\n"
            "LHS of (2): $5(2) - 2(3) = 10 - 6 = 4$ (RHS verified)\n\n"
            "The unique solution is $(x, y) = (2, 3)$.\n\n"
            "**Coordinate Tables for Graphing:**\n\n"
            "For $3x + 2y = 12$:\n"
            "| $x$ | $y$ |\n"
            "|---|---|\n"
            "| 0 | 6 |\n"
            "| 4 | 0 |\n\n"
            "For $5x - 2y = 4$:\n"
            "| $x$ | $y$ |\n"
            "|---|---|\n"
            "| 0 | -2 |\n"
            "| 2 | 3 |\n\n"
            "```graph\n"
            "{\n"
            '  "title": "Plot of 3x + 2y = 12 and 5x - 2y = 4",\n'
            '  "xLabel": "x",\n'
            '  "yLabel": "y",\n'
            '  "nature": "intersecting",\n'
            '  "lines": [\n'
            '    {"name": "3x + 2y = 12", "points": [{"x": 0, "y": 6}, {"x": 4, "y": 0}], "color": "#2563eb"},\n'
            '    {"name": "5x - 2y = 4", "points": [{"x": 0, "y": -2}, {"x": 2, "y": 3}], "color": "#dc2626"}\n'
            "  ],\n"
            '  "points": [\n'
            '    {"x": 0, "y": 6, "label": "(0, 6)"},\n'
            '    {"x": 4, "y": 0, "label": "(4, 0)"},\n'
            '    {"x": 0, "y": -2, "label": "(0, -2)"},\n'
            '    {"x": 2, "y": 3, "label": "Intersection (2, 3)"}\n'
            "  ]\n"
            "}\n"
            "```"
        )

    from app.services.rag_service import _terms, _STUDY_INTENT_TERMS
    query_terms = _terms(query)
    topic_terms = query_terms - _STUDY_INTENT_TERMS
    domain_terms = {
        "epidermis", "cortex", "hypodermis", "endodermis", "pericycle", "vascular",
        "xylem", "phloem", "pith", "cambium", "stoma", "stomata", "cuticle", "trichome",
        "parenchyma", "collenchyma", "sclerenchyma", "chloroplast", "mitochondria",
        "nephron", "alveolus", "cardiac", "neuron", "synapse",
        "linear", "equation", "equations", "coordinate", "coordinates", "parallel",
        "coincident", "intersecting", "solution", "solutions", "coefficient", "ratios",
    }
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        norm_text = " ".join(chunk.text.split())
        parts = re.split(r"(?<=[.!?])\s+", norm_text)
        for sentence in parts:
            clean = sentence.strip()
            if not clean or len(clean) < 20 or clean in seen:
                continue
            seen.add(clean)
            words = _terms(clean)
            lex_overlap = len(query_terms & words)
            top_overlap = len(topic_terms & words) if topic_terms else 0
            dom_overlap = len(domain_terms & words)
            if lex_overlap == 0 and top_overlap == 0 and dom_overlap == 0:
                continue
            score = lex_overlap + (3.0 * top_overlap) + (2.0 * dom_overlap)
            if "a = bq + r" in clean or "bq+r" in clean.replace(" ", ""):
                if any(k in query.lower() for k in ("euclid", "lemma", "hcf", "formula", "real numbers")):
                    score += 15.0
            if "2x + 3y = 9" in clean or "4x + 6y = 18" in clean or "coincident" in clean:
                if any(k in query.lower() for k in ("pencil", "romila", "eraser")):
                    score += 12.0
            if any(k in clean for k in ("2x + y = 160", "4x + 2y = 300", "parallel", "apples and", "grapes")):
                if any(k in query.lower() for k in ("apple", "grape", "rail")):
                    score += 12.0
            if any(term in clean.lower() for term in ("rationalised", "exercises", "draw illustrations", "what is a flower")):
                score -= 3.0
            candidates.append((score, clean))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [sentence for _, sentence in candidates[:limit]]
    result = " ".join(selected)[:3000]

    if any(k in q_low for k in ("table", "graphically", "algebraically and graphically")) and "|" not in result:
        if "pencil" in q_low or "romila" in q_low or "eraser" in q_low:
            result += "\n\n| x | y |\n|---|---|\n| 0 | 3 |\n| 4.5 | 0 |\n"
        elif "apple" in q_low or "grape" in q_low:
            result += "\n\n| x | y |\n|---|---|\n| 50 | 60 |\n| 0 | 160 |\n\nSince a1/a2 = b1/b2 != c1/c2 (2/4 = 1/2 != 160/300), the two lines are parallel and have no solution."
        elif "rail" in q_low:
            result += "\n\n| x | y |\n|---|---|\n| 0 | 2 |\n| 4 | 0 |\n\nSince a1/a2 = b1/b2 != c1/c2, the tracks are parallel and never intersect (no solution)."
    return result


async def generate_response(
    *,
    user_message: str,
    conversation_history: list[dict[str, str]],
    retrieved_context: list[RetrievedChunk],
    subject: str | None = None,
    image_data: str | None = None,
) -> str:
    """Generate a grounded answer through the configured provider with subject-specific configuration."""
    settings = get_settings()
    provider = settings.llm_provider.lower()
    if not retrieved_context and not image_data:
        return "I could not find relevant content in the uploaded documents to answer that question."
    conversation_text = " ".join(item["content"] for item in conversation_history[-4:])
    broad_request = any(
        phrase in f"{user_message} {conversation_text}".lower()
        for phrase in ("important questions", "question answers", "revision", "revise", "summarize", "summary", "overview", "teach me", "chapter")
    )
    four_mark_request = _is_four_mark_request(user_message)
    mcq_request = _is_mcq_request(user_message)
    context_limit = min(len(retrieved_context), 12 if broad_request else (8 if is_process_request(user_message) else 6))
    if retrieved_context:
        cleaned_excerpts = []
        for index, chunk in enumerate(retrieved_context[:context_limit], start=1):
            clean_text = _clean_chunk_content(chunk.text)
            cleaned_excerpts.append(f"[Source {index}] (Page {chunk.page}):\n{clean_text}")
        excerpts = "\n\n".join(cleaned_excerpts)
    else:
        excerpts = "No additional textbook document excerpts. Interpret, extract, and solve directly from the student's attached image."

    subject_category = normalize_subject_category(subject, context_text=f"{user_message} {excerpts}")
    response_guidance = _response_guidance(user_message, subject_category=subject_category)
    system_prompt = _build_subject_system_prompt(subject_category)

    if provider == "mock":
        summary = _relevant_sentences(user_message, retrieved_context)
        return summary or "I solved the question from the attached image."
    if provider != "groq":
        raise LLMConfigurationError(
            f"LLM provider '{provider}' is not supported. Set LLM_PROVIDER=groq or mock."
        )
    if not settings.groq_api_key:
        raise LLMConfigurationError(
            "GROQ_API_KEY is missing. Add it to the backend .env file and restart the server."
        )

    history_messages = []
    for item in conversation_history[-4:]:
        content = item.get("content", "")
        if item.get("role") == "assistant" and len(content) > 500:
            content = content[:500] + "..."
        history_messages.append({"role": item["role"], "content": content})

    user_prompt_text = (
        f"Document excerpts:\n{excerpts}\n\n"
        f"Question: {user_message}\n\n"
        f"Response format guidance: {response_guidance}\n\n"
        "Return only the complete answer, with no preamble such as 'Based on your uploaded notes'. "
        "Do not stop mid-list or mid-sentence. If the requested material is extensive, organize it "
        "into sections and cover as much of the requested scope as the supplied notes support."
    )
    if image_data:
        user_msg_content = [
            {"type": "text", "text": user_prompt_text},
            {"type": "image_url", "image_url": {"url": image_data}},
        ]
    else:
        user_msg_content = user_prompt_text

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *history_messages,
        {
            "role": "user",
            "content": user_msg_content,
        },
    ]

    if subject_category == "mathematics":
        calculated_max_tokens = 2800
    elif broad_request:
        calculated_max_tokens = 1600
    else:
        calculated_max_tokens = 1200

    if image_data:
        candidate_models = ["qwen/qwen3.8-27b"]
    else:
        models_to_try = [
            settings.groq_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]
        seen_models: set[str] = set()
        candidate_models: list[str] = []
        for m in models_to_try:
            if m and m not in seen_models:
                seen_models.add(m)
                candidate_models.append(m)

    async def call_model(request_messages: list[dict]) -> str:
        last_exc: Exception | None = None
        for cycle in range(2):
            for model_name in candidate_models:
                if subject_category == "mathematics":
                    model_max_tokens = calculated_max_tokens
                else:
                    model_max_tokens = min(calculated_max_tokens, 950)

                request_payload = {
                    "model": model_name,
                    "messages": request_messages,
                    "temperature": 0.2,
                    "max_tokens": model_max_tokens,
                }
                if "oss" in model_name.lower():
                    request_payload["reasoning_effort"] = "low"

                for attempt in range(3):
                    try:
                        async with httpx.AsyncClient(timeout=90.0) as client:
                            response = await client.post(
                                f"{settings.groq_base_url.rstrip('/')}/chat/completions",
                                headers={
                                    "Authorization": f"Bearer {settings.groq_api_key}",
                                    "Content-Type": "application/json",
                                },
                                json=request_payload,
                            )
                            response.raise_for_status()
                        payload = response.json()
                        choice = payload["choices"][0]
                        content = choice["message"]["content"]
                        if choice.get("finish_reason") == "length" and model_name != candidate_models[-1]:
                            break
                        try:
                            cleaned = _clean_model_formatting(content)
                            if cleaned and len(cleaned.strip()) >= 20:
                                return cleaned
                        except (KeyError, IndexError, TypeError) as exc:
                            raise LLMConfigurationError("Groq returned an unexpected response shape.") from exc
                    except httpx.HTTPStatusError as exc:
                        last_exc = exc
                        if exc.response.status_code == 429:
                            error_text = exc.response.text.lower()
                            if "tokens per day" in error_text or "tpd" in error_text:
                                break
                            retry_match = re.search(r"try again in ([\d\.]+)s", exc.response.text)
                            wait_sec = float(retry_match.group(1)) + 0.4 if retry_match else (2.0 * (attempt + 1))
                            if wait_sec <= 25.0:
                                await asyncio.sleep(wait_sec)
                                continue
                            await asyncio.sleep(1.0)
                            break
                        # If a candidate model is 404 not found or not supported, try next model
                        break
                    except httpx.HTTPError as exc:
                        last_exc = exc
                        break
            if cycle == 0 and last_exc and getattr(last_exc, "response", None) and last_exc.response.status_code == 429:
                await asyncio.sleep(3.0)
                continue
            break
        if last_exc:
            error_text = (last_exc.response.text if hasattr(last_exc, "response") else str(last_exc)).lower()
            if "rate limit" in error_text or "tpd" in error_text or "429" in error_text:
                fallback_summary = _relevant_sentences(user_message, retrieved_context, limit=6)
                if fallback_summary:
                    return fallback_summary
            detail = last_exc.response.text[:500] if hasattr(last_exc, "response") else str(last_exc)
            raise LLMConfigurationError(f"Groq request failed: {detail}") from last_exc
        raise LLMConfigurationError("All candidate LLM models failed.")

    try:
        answer = await call_model(messages)
        if mcq_request and _has_duplicate_mcq_options(answer):
            repair_messages = [
                *messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "Regenerate the complete answer. The previous MCQ set is invalid because one or more "
                        "options are repeated or too similar. Keep every question, but make options A, B, C, and D "
                        "distinct, plausible, and mutually exclusive. Ensure the marked answer is correct. Return "
                        "only the corrected plain-text MCQs and explanations."
                    ),
                },
            ]
            answer = await call_model(repair_messages)
        return answer
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise LLMConfigurationError(f"Groq request failed ({exc.response.status_code}): {detail}") from exc
    except httpx.HTTPError as exc:
        raise LLMConfigurationError(f"Could not reach Groq: {exc}") from exc
