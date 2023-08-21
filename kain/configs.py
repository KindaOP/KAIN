from collections import OrderedDict
from typing import Sequence, Tuple, Dict


class General:
	#################### CUSTOMISABLE ####################
	NUM_BLOCKS_PER_ENC = 3
	NUM_HEADS_PER_BLOCK = 2
	NUM_FEEDFORWARD_DIM = 256
	ADDNORM_DROPOUT_RATE = 0.1
	######################################################

	NUM_FEATURES = None


class Text:
	#################### CUSTOMISABLE ####################
	MAX_CHARS_PER_SENT = 196
	######################################################
	
	PAD_TOKEN = "<PAD>"
	UNICODES = OrderedDict({
		'Basic Latin': (0, 127),
		'Latin Extended-A': (256, 383),
		'IPA Extensions': (592, 687),
		'Combining Diacritical Marks': (768, 879),
		'Cyrillic': (1024, 1279),
		'Armenian': (1328, 1423),
		'Arabic': (1536, 1791),
		'Arabic Supplement': (1872, 1919),
		'NKo': (1984, 2047),
		'Mandaic': (2112, 2143),
		'Arabic Extended-B': (2160, 2207),
		'Devanagari': (2304, 2431),
		'Gurmukhi': (2560, 2687),
		'Oriya': (2816, 2943),
		'Telugu': (3072, 3199),
		'Malayalam': (3328, 3455),
		'Thai': (3584, 3711),
		'Tibetan': (3840, 4095),
		'Georgian': (4256, 4351),
		'Ethiopic': (4608, 4991),
		'Cherokee': (5024, 5119),
		'Ogham': (5760, 5791),
		'Tagalog': (5888, 5919),
		'Buhid': (5952, 5983),
		'Khmer': (6016, 6143),
		'Unified Canadian Aboriginal Syllabics Extended': (6320, 6399),
		'Tai Le': (6480, 6527),
		'Khmer Symbols': (6624, 6655),
		'Tai Tham': (6688, 6831),
		'Balinese': (6912, 7039),
		'Batak': (7104, 7167),
		'Ol Chiki': (7248, 7295),
		'Georgian Extended': (7312, 7359),
		'Vedic Extensions': (7376, 7423),
		'Phonetic Extensions Supplement': (7552, 7615),
		'Latin Extended Additional': (7680, 7935),
		'General Punctuation': (8192, 8303),
		'Currency Symbols': (8352, 8399),
		'Letterlike Symbols': (8448, 8527),
		'Arrows': (8592, 8703),
		'Miscellaneous Technical': (8960, 9215),
		'Optical Character Recognition': (9280, 9311),
		'Box Drawing': (9472, 9599),
		'Geometric Shapes': (9632, 9727),
		'Dingbats': (9984, 10175),
		'Supplemental Arrows-A': (10224, 10239),
		'Supplemental Arrows-B': (10496, 10623),
		'Supplemental Mathematical Operators': (10752, 11007),
		'Glagolitic': (11264, 11359),
		'Coptic': (11392, 11519),
		'Tifinagh': (11568, 11647),
		'Cyrillic Extended-A': (11744, 11775),
		'CJK Radicals Supplement': (11904, 12031),
		'Ideographic Description Characters': (12272, 12287),
		'Hiragana': (12352, 12447),
		'Bopomofo': (12544, 12591),
		'Kanbun': (12688, 12703),
		'CJK Strokes': (12736, 12783),
		'Enclosed CJK Letters and Months': (12800, 13055),
		'CJK Unified Ideographs Extension A': (13312, 19903),
		'CJK Unified Ideographs': (19968, 40959),
		'Yi Radicals': (42128, 42191),
		'Vai': (42240, 42559),
		'Bamum': (42656, 42751),
		'Latin Extended-D': (42784, 43007),
		'Common Indic Number Forms': (43056, 43071),
		'Saurashtra': (43136, 43231),
		'Kayah Li': (43264, 43311),
		'Hangul Jamo Extended-A': (43360, 43391),
		'Myanmar Extended-B': (43488, 43519),
		'Myanmar Extended-A': (43616, 43647),
		'Meetei Mayek Extensions': (43744, 43775),
		'Latin Extended-E': (43824, 43887),
		'Meetei Mayek': (43968, 44031),
		'Hangul Jamo Extended-B': (55216, 55295),
		'High Private Use Surrogates': (56192, 56319),
		'Private Use Area': (57344, 63743),
		'Alphabetic Presentation Forms': (64256, 64335),
		'Variation Selectors': (65024, 65039),
		'Combining Half Marks': (65056, 65071),
		'Small Form Variants': (65104, 65135),
		'Halfwidth and Fullwidth Forms': (65280, 65519),
		'Linear B Syllabary': (65536, 65663),
		'Aegean Numbers': (65792, 65855),
		'Ancient Symbols': (65936, 65999),
		'Lycian': (66176, 66207),
		'Coptic Epact Numbers': (66272, 66303),
		'Gothic': (66352, 66383),
		'Ugaritic': (66432, 66463),
		'Deseret': (66560, 66639),
		'Osmanya': (66688, 66735),
		'Elbasan': (66816, 66863),
		'Vithkuqi': (66928, 67007),
		'Latin Extended-F': (67456, 67519),
		'Imperial Aramaic': (67648, 67679),
		'Nabataean': (67712, 67759),
		'Phoenician': (67840, 67871),
		'Meroitic Hieroglyphs': (67968, 67999),
		'Kharoshthi': (68096, 68191),
		'Old North Arabian': (68224, 68255),
		'Avestan': (68352, 68415),
		'Inscriptional Pahlavi': (68448, 68479),
		'Old Turkic': (68608, 68687),
		'Hanifi Rohingya': (68864, 68927),
		'Yezidi': (69248, 69311),
		'Old Sogdian': (69376, 69423),
		'Old Uyghur': (69488, 69551),
		'Elymaic': (69600, 69631),
		'Kaithi': (69760, 69839),
		'Chakma': (69888, 69967),
		'Sharada': (70016, 70111),
		'Khojki': (70144, 70223),
		'Khudawadi': (70320, 70399),
		'Newa': (70656, 70783),
		'Siddham': (71040, 71167),
		'Mongolian Supplement': (71264, 71295),
		'Ahom': (71424, 71503),
		'Warang Citi': (71840, 71935),
		'Nandinagari': (72096, 72191),
		'Soyombo': (72272, 72367),
		'Pau Cin Hau': (72384, 72447),
		'Bhaiksuki': (72704, 72815),
		'Masaram Gondi': (72960, 73055),
		'Makasar': (73440, 73471),
		'Lisu Supplement': (73648, 73663),
		'Cuneiform': (73728, 74751),
		'Early Dynastic Cuneiform': (74880, 75087),
		'Egyptian Hieroglyphs': (77824, 78895),
		'Anatolian Hieroglyphs': (82944, 83583),
		'Mro': (92736, 92783),
		'Bassa Vah': (92880, 92927),
		'Medefaidrin': (93760, 93855),
		'Ideographic Symbols and Punctuation': (94176, 94207),
		'Tangut Components': (100352, 101119),
		'Tangut Supplement': (101632, 101759),
		'Kana Supplement': (110592, 110847),
		'Small Kana Extension': (110896, 110959),
		'Duployan': (113664, 113823),
		'Znamenny Musical Notation': (118528, 118735),
		'Musical Symbols': (119040, 119295),
		'Kaktovik Numerals': (119488, 119519),
		'Tai Xuan Jing Symbols': (119552, 119647),
		'Mathematical Alphanumeric Symbols': (119808, 120831),
		'Latin Extended-G': (122624, 122879),
		'Cyrillic Extended-D': (122928, 123023),
		'Toto': (123536, 123583),
		'Nag Mundari': (124112, 124159),
		'Mende Kikakui': (124928, 125151),
		'Indic Siyaq Numbers': (126064, 126143),
		'Arabic Mathematical Alphabetic Symbols': (126464, 126719),
		'Domino Tiles': (127024, 127135),
		'Enclosed Alphanumeric Supplement': (127232, 127487),
		'Miscellaneous Symbols and Pictographs': (127744, 128511),
		'Ornamental Dingbats': (128592, 128639),
		'Alchemical Symbols': (128768, 128895),
		'Supplemental Arrows-C': (129024, 129279),
		'Chess Symbols': (129536, 129647),
		'Symbols for Legacy Computing': (129792, 130047),
		'CJK Unified Ideographs Extension C': (173824, 177983),
		'CJK Unified Ideographs Extension E': (178208, 183983),
		'CJK Compatibility Ideographs Supplement': (194560, 195103),
		'CJK Unified Ideographs Extension H': (201552, 205743),
		'Variation Selectors Supplement': (917760, 917999),
		'Supplementary Private Use Area-B': (1048576, 1114111),
	})
        
	@staticmethod
	def get_character_dicts(
        langs:Sequence[str]
        ) -> Tuple[Dict[str, int], Dict[int, str]]:
		char_list = [Text.PAD_TOKEN]
		for name, (ci, cf) in Text.UNICODES.items():
			if name in langs:
				char_list += [chr(c) for c in range(ci, cf+1, 1)]
		encoding_dict = {}
		decoding_dict = {}
		for i, c in enumerate(char_list):
			encoding_dict[c] = i
			decoding_dict[i] = c
		return encoding_dict, decoding_dict


class Image:
	#################### CUSTOMISABLE ####################
	IMAGE_SHAPE = (224, 224)
	NUM_PATCHES = (28, 28)
	NUM_CHANNELS = 3
	######################################################
	
	General.NUM_FEATURES = (
		IMAGE_SHAPE[0]//NUM_PATCHES[0] * \
		IMAGE_SHAPE[1]//NUM_PATCHES[1] * \
		NUM_CHANNELS
	)


class Voice:
    #################### CUSTOMISABLE ####################
	SAMPLING_RATE = 12000
	MAX_SECONDS = 5
	######################################################
	
	MAX_SIGNAL_LENGTH = SAMPLING_RATE * MAX_SECONDS

