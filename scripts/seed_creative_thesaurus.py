"""
Seed script: populate writing_thesaurus collection with creative writing clichés to avoid.

Covers EN and PT vocabulary for poetry, songwriting, and prose fiction domains.

Usage:
    cd /path/to/mcp-writing-library
    source .venv/bin/activate
    python scripts/seed_creative_thesaurus.py [--no-enrich]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tools.collections import setup_collections
from src.tools.thesaurus import add_thesaurus_entry

# ---------------------------------------------------------------------------
# English creative writing vocabulary
# ---------------------------------------------------------------------------

CREATIVE_EN_WORDS = [
    # --- Poetry domain ---
    {
        "headword": "muse",
        "language": "en",
        "domain": "poetry",
        "part_of_speech": "noun",
        "register": "poetic",
        "why_avoid": "Romantic-era abstraction that deflects agency. Implies inspiration arrives passively rather than being crafted through attention and revision.",
        "alternatives": [
            {"word": "source", "meaning_nuance": "The actual material: memory, observation, or experience", "register": "neutral", "when_to_use": "When describing where the poem originates"},
            {"word": "impulse", "meaning_nuance": "The urgency or pressure behind the work", "register": "neutral", "when_to_use": "When describing creative motivation"},
            {"word": "subject", "meaning_nuance": "What the poem is actually about", "register": "neutral", "when_to_use": "When naming the poem's focus"},
        ],
        "collocations": ["my muse visited me", "muse abandoned me", "waiting for the muse"],
        "example_bad": "My muse visited me this morning and the poem wrote itself.",
        "example_good": "I kept returning to my grandmother's hands until the poem found its shape.",
    },
    {
        "headword": "tapestry",
        "language": "en",
        "domain": "poetry",
        "part_of_speech": "noun",
        "register": "poetic",
        "why_avoid": "Overused metaphor for complexity or interconnection. Signals the writer reached for the nearest available image rather than finding a specific one.",
        "alternatives": [
            {"word": "web", "meaning_nuance": "Interconnected but potentially trapping", "register": "neutral", "when_to_use": "When entanglement or risk is part of the meaning"},
            {"word": "braid", "meaning_nuance": "Multiple strands woven together deliberately", "register": "neutral", "when_to_use": "When the joining is intentional and skilled"},
        ],
        "collocations": ["tapestry of emotions", "rich tapestry", "tapestry of life"],
        "example_bad": "Her life was a rich tapestry of joy and sorrow.",
        "example_good": "Her life had been mended so many times the patches were its pattern.",
    },
    {
        "headword": "resonate",
        "language": "en",
        "domain": "poetry",
        "part_of_speech": "verb",
        "register": "formal",
        "why_avoid": "Vague critical vocabulary that tells the reader how to feel rather than achieving the effect. Overused in workshop and review contexts.",
        "alternatives": [
            {"word": "stay with", "meaning_nuance": "The image or line lingers after reading", "register": "neutral", "when_to_use": "When describing lasting emotional impact"},
            {"word": "return to", "meaning_nuance": "The reader comes back to it involuntarily", "register": "neutral", "when_to_use": "When describing a compelling image"},
        ],
        "collocations": ["deeply resonates", "resonates with readers", "resonant imagery"],
        "example_bad": "The final stanza resonates with profound emotional depth.",
        "example_good": "I kept rereading the final stanza — the word 'nevertheless' doing all the work.",
    },
    {
        "headword": "evocative",
        "language": "en",
        "domain": "poetry",
        "part_of_speech": "adjective",
        "register": "formal",
        "why_avoid": "Tells the reader the poem evokes something without showing what or how. Empty praise word when used in workshop or self-description.",
        "alternatives": [
            {"word": "specific", "meaning_nuance": "The image works because it is exact, not general", "register": "neutral", "when_to_use": "When praising concrete detail"},
            {"word": "sensory", "meaning_nuance": "Engages sight, sound, smell, touch, or taste", "register": "neutral", "when_to_use": "When describing multi-sense imagery"},
        ],
        "collocations": ["evocative imagery", "highly evocative", "evocative language"],
        "example_bad": "The poem uses highly evocative imagery to convey loss.",
        "example_good": "The poem renders loss through a single unwashed cup left on the draining board.",
    },
    # --- Songwriting domain ---
    {
        "headword": "heartstrings",
        "language": "en",
        "domain": "songwriting",
        "part_of_speech": "noun",
        "register": "lyrical",
        "why_avoid": "Clichéd metaphor for emotional connection. Signals lazy lyric-writing; the emotion is named rather than created.",
        "alternatives": [
            {"word": "chest", "meaning_nuance": "Physical location of emotional sensation", "register": "lyrical", "when_to_use": "When the physical sensation of emotion is the point"},
            {"word": "pull", "meaning_nuance": "The gravitational feeling of longing", "register": "lyrical", "when_to_use": "When describing involuntary emotional draw"},
        ],
        "collocations": ["tugs at heartstrings", "pull at my heartstrings", "heartstrings connected"],
        "example_bad": "Your voice pulls at my heartstrings every time.",
        "example_good": "Your voice puts weight in my chest I can't set down.",
    },
    {
        "headword": "bittersweet",
        "language": "en",
        "domain": "songwriting",
        "part_of_speech": "adjective",
        "register": "lyrical",
        "why_avoid": "Abstract emotional label that does the reader's work for them. Does not evoke the specific texture of mixed emotions.",
        "alternatives": [
            {"word": "laughing through it", "meaning_nuance": "Specific action showing mixed emotion", "register": "lyrical", "when_to_use": "When joy and pain coexist in visible behaviour"},
            {"word": "glad and wrong about it", "meaning_nuance": "Captures internal contradiction specifically", "register": "lyrical", "when_to_use": "When the speaker knows their feeling is complicated"},
        ],
        "collocations": ["bittersweet memories", "bittersweet feeling", "bittersweet goodbye"],
        "example_bad": "This goodbye feels bittersweet to me.",
        "example_good": "I'm smiling at the airport and I hate myself for it.",
    },
    {
        "headword": "journey",
        "language": "en",
        "domain": "songwriting",
        "part_of_speech": "noun",
        "register": "lyrical",
        "why_avoid": "Overused metaphor for life experience or personal growth. Has lost all specificity through repetition.",
        "alternatives": [
            {"word": "road", "meaning_nuance": "Literal path with physical texture", "register": "lyrical", "when_to_use": "When the physical metaphor of travel is deliberately invoked"},
            {"word": "it", "meaning_nuance": "Refusing to name the experience can be more powerful", "register": "lyrical", "when_to_use": "When the experience is better shown than labelled"},
        ],
        "collocations": ["this journey we're on", "life journey", "our journey together"],
        "example_bad": "This journey we've been on has changed us both.",
        "example_good": "We've changed each other in ways we can't undo.",
    },
    # --- Fiction domain ---
    {
        "headword": "penned",
        "language": "en",
        "domain": "fiction",
        "part_of_speech": "verb",
        "register": "formal",
        "why_avoid": "Precious synonym for 'wrote'. Signals self-conscious literariness in critical or author-facing writing.",
        "alternatives": [
            {"word": "wrote", "meaning_nuance": "Direct and unpretentious", "register": "neutral", "when_to_use": "Always"},
            {"word": "composed", "meaning_nuance": "Appropriate for formal or musical works", "register": "formal", "when_to_use": "Letters, formal essays, musical scores"},
        ],
        "collocations": ["penned her debut novel", "penned a letter", "the author has penned"],
        "example_bad": "She penned her debut novel in three months.",
        "example_good": "She wrote her debut novel in three months.",
    },
    {
        "headword": "tome",
        "language": "en",
        "domain": "fiction",
        "part_of_speech": "noun",
        "register": "formal",
        "why_avoid": "Pompous synonym for 'book'. Signals affected literary self-consciousness when used in marketing or description.",
        "alternatives": [
            {"word": "book", "meaning_nuance": "Direct and universal", "register": "neutral", "when_to_use": "Always"},
            {"word": "novel", "meaning_nuance": "Specific to the form", "register": "neutral", "when_to_use": "When the form is a novel"},
        ],
        "collocations": ["weighty tome", "massive tome", "the tome sat unread"],
        "example_bad": "The weighty tome sat unread on her shelf.",
        "example_good": "The book sat unread on her shelf.",
    },
    {
        "headword": "wordsmith",
        "language": "en",
        "domain": "fiction",
        "part_of_speech": "noun",
        "register": "informal",
        "why_avoid": "Clichéd self-description for writers. Implies craft through the tool rather than the mind; overused in bios and marketing.",
        "alternatives": [
            {"word": "writer", "meaning_nuance": "Direct, no affectation", "register": "neutral", "when_to_use": "Always"},
            {"word": "author", "meaning_nuance": "For published work contexts", "register": "formal", "when_to_use": "When publication is the context"},
        ],
        "collocations": ["skilled wordsmith", "a true wordsmith", "wordsmith extraordinaire"],
        "example_bad": "A skilled wordsmith, she has crafted three bestselling novels.",
        "example_good": "She has written three bestselling novels.",
    },
    {
        "headword": "whispers of",
        "language": "en",
        "domain": "fiction",
        "part_of_speech": "phrase",
        "register": "poetic",
        "why_avoid": "Overused atmospheric phrase that substitutes vague suggestion for specific image. Signals the writer avoided committing to a concrete detail.",
        "alternatives": [
            {"word": "traces of", "meaning_nuance": "Physical evidence, not impression", "register": "neutral", "when_to_use": "When something leaves a physical mark"},
            {"word": "faint [specific noun]", "meaning_nuance": "Name the actual thing faintly sensed", "register": "neutral", "when_to_use": "When the object of perception can be named"},
        ],
        "collocations": ["whispers of autumn", "whispers of sadness", "whispers of the past"],
        "example_bad": "The room held whispers of a life once lived.",
        "example_good": "The room still smelled of the tobacco she had quit twenty years ago.",
    },
    {
        "headword": "hauntingly beautiful",
        "language": "en",
        "domain": "fiction",
        "part_of_speech": "phrase",
        "register": "formal",
        "why_avoid": "Stock praise phrase that tells rather than shows. Doing both at once neutralises both.",
        "alternatives": [
            {"word": "describe the specific quality", "meaning_nuance": "Name what makes it haunting or beautiful specifically", "register": "neutral", "when_to_use": "Always"},
        ],
        "collocations": ["hauntingly beautiful prose", "hauntingly beautiful voice", "hauntingly beautiful image"],
        "example_bad": "Her prose is hauntingly beautiful.",
        "example_good": "Her prose makes you feel you have forgotten something important.",
    },
]

# ---------------------------------------------------------------------------
# Portuguese creative writing vocabulary
# ---------------------------------------------------------------------------

CREATIVE_PT_WORDS = [
    # --- Poetry domain ---
    {
        "headword": "versejar",
        "language": "pt",
        "domain": "poetry",
        "part_of_speech": "verb",
        "register": "poetic",
        "why_avoid": "Arcaico e afectado. Implica poesia mecânica em vez de escrita intencional e trabalhada.",
        "alternatives": [
            {"word": "escrever", "meaning_nuance": "Directo e sem afectação", "register": "neutral", "when_to_use": "Sempre"},
            {"word": "compor", "meaning_nuance": "Adequado para obras mais formais ou musicais", "register": "formal", "when_to_use": "Poesia formal, poesia musicada"},
        ],
        "collocations": ["versejar sobre o amor", "versejar tristezas", "habilidade de versejar"],
        "example_bad": "Passou a vida a versejar sobre a saudade.",
        "example_good": "Passou a vida a escrever sobre a saudade.",
    },
    {
        "headword": "tapeçaria de emoções",
        "language": "pt",
        "domain": "poetry",
        "part_of_speech": "phrase",
        "register": "poetic",
        "why_avoid": "Metáfora gasta que nomeia a complexidade emocional em vez de a criar. Sinal de que a imagem específica foi evitada.",
        "alternatives": [
            {"word": "teia", "meaning_nuance": "Ligação que pode também aprisionar", "register": "neutral", "when_to_use": "Quando o enredamento ou o risco fazem parte do sentido"},
            {"word": "entrelaçamento", "meaning_nuance": "Cruzamento de fios distintos com intenção", "register": "neutral", "when_to_use": "Quando a junção é deliberada e visível"},
        ],
        "collocations": ["rica tapeçaria de emoções", "tapeçaria da vida", "tapeçaria de sentimentos"],
        "example_bad": "A sua poesia é uma rica tapeçaria de emoções.",
        "example_good": "A sua poesia transforma a memória em peso físico.",
    },
    {
        "headword": "ressoa",
        "language": "pt",
        "domain": "poetry",
        "part_of_speech": "verb",
        "register": "formal",
        "why_avoid": "Vocabulário crítico vago que diz ao leitor o que sentir em vez de criar o efeito. Sobrepõe-se ao trabalho do poema.",
        "alternatives": [
            {"word": "fica connosco", "meaning_nuance": "A imagem ou verso permanece depois da leitura", "register": "neutral", "when_to_use": "Quando se descreve impacto emocional duradouro"},
            {"word": "volta à mente", "meaning_nuance": "O leitor regressa involuntariamente", "register": "neutral", "when_to_use": "Quando se descreve uma imagem que persiste"},
        ],
        "collocations": ["ressoa profundamente", "ressoa com o leitor", "imagem que ressoa"],
        "example_bad": "A estrofe final ressoa com profunda emoção.",
        "example_good": "Continuo a relê-la — a palavra 'apesar' a fazer todo o trabalho.",
    },
    # --- Songwriting domain ---
    {
        "headword": "doce melodia",
        "language": "pt",
        "domain": "songwriting",
        "part_of_speech": "phrase",
        "register": "lyrical",
        "why_avoid": "Clichê de letra de música. Abstracção sem imagem concreta que não evoca qualquer sensação específica.",
        "alternatives": [
            {"word": "a sua voz a roubar o silêncio", "meaning_nuance": "Imagem concreta da presença vocal", "register": "lyrical", "when_to_use": "Quando a voz do intérprete é o assunto"},
            {"word": "o som que fica quando callas", "meaning_nuance": "A ausência como presença", "register": "lyrical", "when_to_use": "Quando o silêncio após a música é o que importa"},
        ],
        "collocations": ["doce melodia que ressoa", "doce melodia no coração", "ouvir a doce melodia"],
        "example_bad": "A doce melodia ecoou pelo quarto.",
        "example_good": "A sua voz entrou pela janela sem pedir licença.",
    },
    {
        "headword": "agridoce",
        "language": "pt",
        "domain": "songwriting",
        "part_of_speech": "adjective",
        "register": "lyrical",
        "why_avoid": "Rótulo emocional abstracto que faz o trabalho do leitor em vez de criar a sensação de emoções mistas.",
        "alternatives": [
            {"word": "a rir e a sangrar", "meaning_nuance": "Acção específica que mostra emoção mista", "register": "lyrical", "when_to_use": "Quando alegria e dor coexistem visivelmente"},
            {"word": "feliz e errado nisso", "meaning_nuance": "Captura a contradição interna", "register": "lyrical", "when_to_use": "Quando o locutor sabe que o seu sentimento é complicado"},
        ],
        "collocations": ["memória agridoce", "sensação agridoce", "despedida agridoce"],
        "example_bad": "Esta despedida tem um sabor agridoce.",
        "example_good": "Estou a sorrir no aeroporto e odeio-me por isso.",
    },
    {
        "headword": "jornada",
        "language": "pt",
        "domain": "songwriting",
        "part_of_speech": "noun",
        "register": "lyrical",
        "why_avoid": "Metáfora gasta para experiência de vida ou crescimento pessoal. Perdeu toda a especificidade pela repetição.",
        "alternatives": [
            {"word": "caminho", "meaning_nuance": "Trajecto com textura física", "register": "lyrical", "when_to_use": "Quando a metáfora física da viagem é deliberada"},
            {"word": "isso tudo", "meaning_nuance": "Recusar nomear pode ser mais poderoso", "register": "lyrical", "when_to_use": "Quando a experiência se mostra melhor do que se nomeia"},
        ],
        "collocations": ["nesta jornada juntos", "jornada de vida", "jornada interior"],
        "example_bad": "Nesta jornada que percorremos juntos, mudámos.",
        "example_good": "Mudámo-nos um ao outro de formas que não têm volta.",
    },
    # --- Fiction domain ---
    {
        "headword": "plasmou",
        "language": "pt",
        "domain": "fiction",
        "part_of_speech": "verb",
        "register": "formal",
        "why_avoid": "Sinónimo afectado de 'escreveu' ou 'criou'. Sinaliza auto-consciência literária excessiva em contextos editoriais ou de marketing.",
        "alternatives": [
            {"word": "escreveu", "meaning_nuance": "Directo e sem afectação", "register": "neutral", "when_to_use": "Sempre"},
            {"word": "criou", "meaning_nuance": "Quando o acto criativo é o foco", "register": "neutral", "when_to_use": "Quando a originalidade da obra é o ponto"},
        ],
        "collocations": ["plasmou no papel", "plasmou a sua visão", "habilmente plasmou"],
        "example_bad": "Com mestria, plasmou no papel a sua visão do mundo.",
        "example_good": "Com mestria, escreveu a sua visão do mundo.",
    },
    {
        "headword": "sussurros de",
        "language": "pt",
        "domain": "fiction",
        "part_of_speech": "phrase",
        "register": "poetic",
        "why_avoid": "Frase atmosférica gasta que substitui a sugestão vaga pela imagem específica. Sinaliza evitamento do detalhe concreto.",
        "alternatives": [
            {"word": "vestígios de", "meaning_nuance": "Evidência física, não impressão", "register": "neutral", "when_to_use": "Quando algo deixa uma marca física"},
            {"word": "leve cheiro a [substantivo específico]", "meaning_nuance": "Nomear o que se percepciona levemente", "register": "neutral", "when_to_use": "Quando o objecto da percepção pode ser nomeado"},
        ],
        "collocations": ["sussurros de outono", "sussurros de tristeza", "sussurros do passado"],
        "example_bad": "O quarto guardava sussurros de uma vida outrora vivida.",
        "example_good": "O quarto ainda cheirava ao tabaco que ela abandonara vinte anos antes.",
    },
    {
        "headword": "artesão das palavras",
        "language": "pt",
        "domain": "fiction",
        "part_of_speech": "phrase",
        "register": "informal",
        "why_avoid": "Auto-descrição clichê para escritores. Equivalente PT de 'wordsmith'; implica artesanato mecânico em vez de pensamento.",
        "alternatives": [
            {"word": "escritor", "meaning_nuance": "Directo, sem afectação", "register": "neutral", "when_to_use": "Sempre"},
            {"word": "autor", "meaning_nuance": "Para contextos de publicação", "register": "formal", "when_to_use": "Quando a publicação é o contexto"},
        ],
        "collocations": ["verdadeiro artesão das palavras", "habilidoso artesão das palavras"],
        "example_bad": "Um verdadeiro artesão das palavras, escreveu três romances aclamados.",
        "example_good": "Escreveu três romances aclamados pela crítica.",
    },
]


def seed(enrich: bool = True):
    print("Setting up collections...")
    setup_result = setup_collections()
    thesaurus_status = setup_result.get("thesaurus", {}).get("status", "unknown")
    print(f"  writing_thesaurus: {thesaurus_status}")

    all_entries = CREATIVE_EN_WORDS + CREATIVE_PT_WORDS
    print(f"\nSeeding {len(all_entries)} creative thesaurus entries ({len(CREATIVE_EN_WORDS)} EN, {len(CREATIVE_PT_WORDS)} PT)...")

    succeeded = 0
    skipped = 0
    failed = 0

    for entry in all_entries:
        result = add_thesaurus_entry(**entry)
        lang = entry["language"].upper()
        domain = entry["domain"]
        headword = entry["headword"]

        if result.get("success"):
            print(f"  [OK] [{lang} | {domain}] {headword}")
            succeeded += 1
        elif "already exists" in str(result.get("error", "")).lower():
            print(f"  [SKIP] [{lang} | {domain}] {headword} — already exists")
            skipped += 1
        else:
            print(f"  [FAIL] [{lang} | {domain}] {headword} — ERROR: {result.get('error')}")
            failed += 1

    print(f"\nDone. {succeeded} succeeded, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed creative writing thesaurus entries.")
    parser.add_argument("--no-enrich", action="store_true", help="Skip external API enrichment (not used for creative entries, kept for CLI consistency)")
    args = parser.parse_args()
    seed(enrich=not args.no_enrich)
