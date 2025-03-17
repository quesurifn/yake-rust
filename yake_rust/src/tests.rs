use pretty_assertions::assert_eq;

use super::*;

fn test<const T: usize>(text: &str, lang: &str, cfg: Config, n_best: usize, expected: [(&str, &str, f64); T]) {
    let stopwords = StopWords::predefined(lang).unwrap();
    let mut actual = Yake::new(stopwords, cfg).get_n_best(text, n_best);
    // leave only 4 digits
    actual.iter_mut().for_each(|r| r.score = (r.score * 10_000.).round() / 10_000.);
    assert_eq!(actual, expected);
}

#[test]
fn empty_text() {
    test("", "en", Config::default(), 1, []);
}

#[test]
fn zero_size_ngram() {
    test("happy new year", "en", Config { ngrams: 0, ..Default::default() }, 1, []);
}

#[test]
fn short() {
    test("this is a keyword", "en", Config::default(), 1, [("keyword", "keyword", 0.1583)]);
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn keywords_order_is_preserved() {
    // If not, this test becomes unstable.
    test(
        "Machine learning",
        "en",
        Config { ngrams: 1, ..Default::default() },
        3,
        [("Machine", "machine", 0.1583), ("learning", "learning", 0.1583)],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn laptop() {
    test(
        "Do you need an Apple laptop?",
        "en",
        Config { ngrams: 1, ..Default::default() },
        2,
        [("Apple", "apple", 0.1448), ("laptop", "laptop", 0.1583)],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn headphones() {
    test(
        "Do you like headphones? \
            Starting this Saturday, we will be kicking off a huge sale of headphones! \
            If you need headphones, we've got you covered!",
        "en",
        Config { ngrams: 1, ..Default::default() },
        3,
        [("headphones", "headphones", 0.1141), ("Saturday", "saturday", 0.2111), ("Starting", "starting", 0.4096)],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn multi_ngram() {
    test(
        "I will give you a great deal if you just read this!",
        "en",
        Config { ngrams: 2, ..Default::default() },
        1,
        [("great deal", "great deal", 0.0257)],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn singular() {
    test(
        // Weird grammar; to compare with the "plural" test
        "One smartwatch. One phone. Many phone.",
        "en",
        Config { ngrams: 1, ..Default::default() },
        2,
        [("smartwatch", "smartwatch", 0.2025), ("phone", "phone", 0.2474)],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn plural() {
    test(
        "One smartwatch. One phone. Many phones.",
        "en",
        Config { ngrams: 1, ..Default::default() },
        3,
        [("smartwatch", "smartwatch", 0.2025), ("phone", "phone", 0.4949), ("phones", "phones", 0.4949)],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn non_hyphenated() {
    // For comparison with the "hyphenated" test
    test("Truly high tech!", "en", Config { ngrams: 2, ..Default::default() }, 1, [("high tech", "high tech", 0.0494)]);
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn hyphenated() {
    test("Truly high-tech!", "en", Config { ngrams: 2, ..Default::default() }, 1, [("high-tech", "high-tech", 0.1583)]);
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn weekly_newsletter_short() {
    test(
        "This is your weekly newsletter!",
        "en",
        Config { ngrams: 2, ..Default::default() },
        3,
        [
            ("weekly newsletter", "weekly newsletter", 0.0494),
            ("newsletter", "newsletter", 0.1583),
            ("weekly", "weekly", 0.2974),
        ],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn weekly_newsletter_long() {
    test(
        "This is your weekly newsletter! \
        Hundreds of great deals - everything from men's fashion \
        to high-tech drones!",
        "en",
        Config { ngrams: 2, ..Default::default() },
        5,
        [
            ("weekly newsletter", "weekly newsletter", 0.0780),
            ("newsletter", "newsletter", 0.2005),
            ("weekly", "weekly", 0.3607),
            ("great deals", "great deals", 0.4456),
            ("high-tech drones", "high-tech drones", 0.4456),
        ],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn weekly_newsletter_long_with_paragraphs() {
    test(
        "This is your weekly newsletter!\n\n \
        \tHundreds of great deals - everything from men's fashion \n\
        to high-tech drones!",
        "en",
        Config { ngrams: 2, ..Default::default() },
        5,
        [
            ("weekly newsletter", "weekly newsletter", 0.0780),
            ("newsletter", "newsletter", 0.2005),
            ("weekly", "weekly", 0.3607),
            ("great deals", "great deals", 0.4456),
            ("high-tech drones", "high-tech drones", 0.4456),
        ],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn composite_recurring_words_and_bigger_window() {
    test(
        "Machine learning is a growing field. Few research fields grow as much as machine learning grows.",
        "en",
        Config { ngrams: 2, window_size: 2, ..Default::default() },
        5,
        [
            ("Machine learning", "machine learning", 0.1346),
            ("growing field", "growing field", 0.1672),
            ("learning", "learning", 0.2265),
            ("Machine", "machine", 0.2341),
            ("growing", "growing", 0.2799),
        ],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn composite_recurring_words_near_numbers() {
    test(
        "I buy 100 yellow bananas every day. Every night I eat bananas - all but 5 bananas.",
        "en",
        Config { ngrams: 2, ..Default::default() },
        3,
        [("yellow bananas", "yellow bananas", 0.0682), ("buy", "buy", 0.1428), ("yellow", "yellow", 0.1428)],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn composite_recurring_words_near_spelled_out_numbers() {
    // For comparison with "composite_recurring_words_near_numbers" to see if numbers cause
    test(
        "I buy a hundred yellow bananas every day. Every night I eat bananas - all but five bananas.",
        "en",
        Config { ngrams: 2, ..Default::default() },
        3,
        [
            ("hundred yellow", "hundred yellow", 0.0446),
            ("yellow bananas", "yellow bananas", 0.1017),
            ("day", "day", 0.1428),
        ],
    );
    // Results agree with reference implementation LIAAD/yake
}

#[test]
fn with_stopword_in_the_middle() {
    test(
        "Game of Thrones",
        "en",
        Config { remove_duplicates: false, ..Config::default() },
        1,
        [("Game of Thrones", "game of thrones", 0.01380)],
    );
    // Results agree with reference implementation LIAAD/yake
}

mod liaad_yake_samples {
    use super::*;

    #[test]
    fn google_sample_single_ngram() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_google.txt"),
            "en",
            Config { ngrams: 1, ..Default::default() },
            10,
            [
                ("Google", "google", 0.0251),
                ("Kaggle", "kaggle", 0.0273),
                ("data", "data", 0.08),
                ("science", "science", 0.0983),
                ("platform", "platform", 0.124),
                ("service", "service", 0.1316),
                ("acquiring", "acquiring", 0.1511),
                ("learning", "learning", 0.1621),
                ("Goldbloom", "goldbloom", 0.1625),
                ("machine", "machine", 0.1672),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn google_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_google.txt"),
            "en",
            Config::default(),
            10,
            [
                ("Google", "google", 0.0251),
                ("Kaggle", "kaggle", 0.0273),
                ("CEO Anthony Goldbloom", "ceo anthony goldbloom", 0.0483),
                ("data science", "data science", 0.055),
                ("acquiring data science", "acquiring data science", 0.0603),
                ("Google Cloud Platform", "google cloud platform", 0.0746),
                ("data", "data", 0.08),
                ("San Francisco", "san francisco", 0.0914),
                ("Anthony Goldbloom declined", "anthony goldbloom declined", 0.0974),
                ("science", "science", 0.0983),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn gitter_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_gitter.txt"),
            "en",
            Config::default(),
            10,
            [
                ("Gitter", "gitter", 0.0190),
                ("GitLab", "gitlab", 0.0478),
                ("acquires software chat", "acquires software chat", 0.0479),
                ("chat startup Gitter", "chat startup gitter", 0.0512),
                ("software chat startup", "software chat startup", 0.0612),
                ("Gitter chat", "gitter chat", 0.0684),
                ("GitLab acquires software", "gitlab acquires software", 0.0685),
                ("startup", "startup", 0.0783),
                ("software", "software", 0.0879),
                ("code", "code", 0.0879),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn genius_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_genius.txt"),
            "en",
            Config::default(),
            10,
            [
                ("Genius", "genius", 0.0261),
                ("company", "company", 0.0263),
                ("Genius quietly laid", "genius quietly laid", 0.027),
                ("company quietly laid", "company quietly laid", 0.0392),
                ("media company", "media company", 0.0404),
                ("Lehman", "lehman", 0.0412),
                ("quietly laid", "quietly laid", 0.0583),
                ("Tom Lehman told", "tom lehman told", 0.0603),
                ("video", "video", 0.0650),
                ("co-founder Tom Lehman", "co-founder tom lehman", 0.0669),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn german_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_german.txt"),
            "de",
            Config::default(),
            10,
            [
                ("Vereinigten Staaten", "vereinigten staaten", 0.0152), // LIAAD REFERENCE: 0.151
                ("Präsidenten Donald Trump", "präsidenten donald trump", 0.0182),
                ("Donald Trump", "donald trump", 0.0211), // LIAAD REFERENCE: 0.21
                ("trifft Donald Trump", "trifft donald trump", 0.0231), // LIAAD REFERENCE: 0.23
                ("Trump", "trump", 0.0240),
                ("Trumps Finanzminister Steven", "trumps finanzminister steven", 0.0243),
                ("Kanzlerin Angela Merkel", "kanzlerin angela merkel", 0.0275), // LIAAD REFERENCE: 0.273
                ("deutsche Kanzlerin Angela", "deutsche kanzlerin angela", 0.0316), // LIAAD REFERENCE: 0.314
                ("Merkel trifft Donald", "merkel trifft donald", 0.0353),       // LIAAD REFERENCE: 0.351
                ("Exportnation Deutschland", "exportnation deutschland", 0.038), // LIAAD REFERENCE: 0.0379
            ],
        );
        // REASONS FOR DISCREPANCY:
        // - The text contains both "bereit" ("ready") and "bereits" ("already").
        //   While "bereits" is a stopword, "bereit" is not.
        //   LIAAD/yake keeps track of whether a term is a stopword or not
        //   in a key-value mapping, where the key is the term, lowercase, plural-normalized.
        //   (Note that the plural normalization techique used is rarely effective in German.)
        //   Since "bereits" occurs before "bereit" in the text, LIAAD/yake sees it,
        //   recognizes it is a stopword, and stores it under the key "bereit". Later,
        //   when it encounters "bereit" (NOT a stopword), it already has that key in its
        //   mapping so it looks it up and finds that it is a keyword (which it is not).
        //   Meanwhile, yake-rust does not have such a key-value store, so it correctly
        //   recognizes "bereits" as a stopword and "bereit" as a non-stopword. The extra
        //   inclusion of "bereit" in the non-stopwords affects the TF statistics and thus
        //   the frequency contribution to the weights, leading to slightly different scores.
        //
        //   This is technically a bug in the reference implementation caused by the plural
        //   normalization. This small discrepancy is thus acceptable.
        //
    }

    #[test]
    fn dutch_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_nl.txt"),
            "nl",
            Config::default(),
            10,
            [
                ("Vincent van Gogh", "vincent van gogh", 0.0111),
                ("Gogh Museum", "gogh museum", 0.0125),
                ("Gogh", "gogh", 0.0150),
                ("Museum", "museum", 0.0438),
                ("brieven", "brieven", 0.0635),
                ("Vincent", "vincent", 0.0643),
                ("Goghs schilderijen", "goghs schilderijen", 0.1009),
                ("Gogh verging", "gogh verging", 0.1215),
                ("Goghs", "goghs", 0.1651),
                ("schrijven", "schrijven", 0.1704),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn finnish_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_fi.txt"),
            "fi",
            Config::default(),
            10,
            [
                ("Mobile Networks", "mobile networks", 0.0043),
                ("Nokia tekee muutoksia", "nokia tekee muutoksia", 0.0061),
                ("tekee muutoksia organisaatioonsa", "tekee muutoksia organisaatioonsa", 0.0065),
                ("johtokuntaansa vauhdittaakseen yhtiön", "johtokuntaansa vauhdittaakseen yhtiön", 0.0088),
                ("vauhdittaakseen yhtiön strategian", "vauhdittaakseen yhtiön strategian", 0.0088),
                ("yhtiön strategian toteuttamista", "yhtiön strategian toteuttamista", 0.0092),
                ("Networks", "networks", 0.0102),
                ("Networks and Applications", "networks and applications", 0.0113),
                ("strategian toteuttamista Nokia", "strategian toteuttamista nokia", 0.0127),
                ("siirtyy Mobile Networks", "siirtyy mobile networks", 0.0130),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn italian_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_it.txt"),
            "it",
            Config::default(),
            5,
            [
                ("Champions League", "champions league", 0.0390),
                ("Quarti", "quarti", 0.0520),
                ("Atlético Madrid", "atlético madrid", 0.0592),
                ("Ottavi di finale", "ottavi di finale", 0.0646),
                ("Real Madrid", "real madrid", 0.0701),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn french_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_fr.txt"),
            "fr",
            Config::default(),
            10,
            [
                ("dégrade en France", "dégrade en france", 0.0254),
                ("jusque-là uniquement associée", "jusque-là uniquement associée", 0.0504),
                ("sondage Ifop réalisé", "sondage ifop réalisé", 0.0554),
                ("religion se dégrade", "religion se dégrade", 0.091),
                ("France", "france", 0.0941),
                ("l'extrême droite", "l'extrême droite", 0.0997),
                ("sondage Ifop", "sondage ifop", 0.101),
                ("Islam", "islam", 0.1021),
                ("musulmane en France", "musulmane en france", 0.1078),
                ("Allemagne", "allemagne", 0.1086),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn portuguese_sport_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_pt_1.txt"),
            "pt",
            Config::default(),
            10,
            [
                ("seleção brasileira treinará", "seleção brasileira treinará", 0.0072),
                ("seleção brasileira", "seleção brasileira", 0.0100),
                ("Seleção Brasileira visando", "seleção brasileira visando", 0.0192),
                ("Seleção Brasileira encara", "seleção brasileira encara", 0.0344),
                ("brasileira treinará", "brasileira treinará", 0.0373),
                ("Renato Augusto", "renato augusto", 0.0376),
                ("Copa da Rússia", "copa da rússia", 0.0407),
                ("seleção", "seleção", 0.0454),
                ("brasileira", "brasileira", 0.0528),
                ("meia Renato Augusto", "meia renato augusto", 0.0623),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn portuguese_tourism_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_pt_2.txt"),
            "pt",
            Config::default(),
            10,
            [
                ("Alvor", "alvor", 0.0165),
                ("Rio Alvor", "rio alvor", 0.0336),
                ("Ria de Alvor", "ria de alvor", 0.0488),
                ("encantadora vila", "encantadora vila", 0.0575),
                ("Algarve", "algarve", 0.0774),
                ("impressionantes de Portugal", "impressionantes de portugal", 0.0844),
                ("estuário do Rio", "estuário do rio", 0.0907),
                ("vila", "vila", 0.1017),
                ("Ria", "ria", 0.1053),
                ("Oceano Atlântico", "oceano atlântico", 0.1357),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn spanish_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_es.txt"),
            "es",
            Config::default(),
            10,
            [
                ("Guerra Civil Española", "guerra civil española", 0.0032),
                ("Guerra Civil", "guerra civil", 0.0130),
                ("Civil Española", "civil española", 0.0153),
                ("Partido Socialista Obrero", "partido socialista obrero", 0.0283),
                ("empezó la Guerra", "empezó la guerra", 0.0333),
                ("Socialista Obrero Español", "socialista obrero español", 0.0411),
                ("José Castillo", "josé castillo", 0.0426),
                ("Española", "española", 0.0566),
                ("José Antonio Primo", "josé antonio primo", 0.0589),
                ("José Calvo Sotelo", "josé calvo sotelo", 0.0596),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn polish_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_pl.txt"),
            "pl",
            Config::default(),
            10,
            [
                ("franka", "franka", 0.0328),
                ("Geerta Wildersa VVD", "geerta wildersa vvd", 0.0346),
                ("Geerta Wildersa", "geerta wildersa", 0.0399),
                ("kurs franka", "kurs franka", 0.0486),
                ("partii Geerta Wildersa", "partii geerta wildersa", 0.0675),
                ("proc", "proc", 0.0692),
                ("mld", "mld", 0.0724),
                ("Narodowego Banku Szwajcarii", "narodowego banku szwajcarii", 0.0728),
                ("kurs franka poniżej", "kurs franka poniżej", 0.0758),
                ("Wildersa", "wildersa", 0.0765),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn turkish_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_tr.txt"),
            "tr",
            Config::default(),
            10,
            [
                ("OECD", "oecd", 0.0178),
                ("Tek Bakışta Eğitim", "tek bakışta eğitim", 0.0236),
                ("eğitim", "eğitim", 0.0278),
                ("OECD eğitim endeksi", "oecd eğitim endeksi", 0.0323),
                ("OECD ortalamasının", "oecd ortalamasının", 0.0383),
                ("Kalkınma Örgütü'nün", "kalkınma örgütü'nün", 0.045),
                ("Tek Bakışta", "tek bakışta", 0.045),
                ("İşbirliği ve Kalkınma", "i̇şbirliği ve kalkınma", 0.0468),
                ("Türkiye'de", "türkiye'de", 0.0480),
                ("yüksek", "yüksek", 0.0513),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn arabic_sample_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_ar.txt"),
            "ar",
            Config::default(),
            10,
            [
                ("عبد السلام العجيلي", "عبد السلام العجيلي", 0.0105),
                ("اللغة العربية الأربعاء", "اللغة العربية الأربعاء", 0.0139),
                ("عبد النبي اصطيف", "عبد النبي اصطيف", 0.0142),
                ("العجيلي في مرآة", "العجيلي في مرآة", 0.0177),
                ("مرآة النقد المقارن", "مرآة النقد المقارن", 0.0183), // LIAAD REFERENCE: 0.018
                ("السلام العجيلي", "السلام العجيلي", 0.0198),
                ("اللغة العربية", "اللغة العربية", 0.0207),
                ("مرآة النقد", "مرآة النقد", 0.0255), // LIAAD REFERENCE: 0.025
                ("اللغة العربية بدمشق", "اللغة العربية بدمشق", 0.0261),
                ("مجمع اللغة العربية", "مجمع اللغة العربية", 0.0281),
            ],
        );
    }

    #[test]
    fn dataset_text_1_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_data_1.txt"),
            "pt",
            Config::default(),
            10,
            [
                ("Médio Oriente continua", "médio oriente continua", 0.0008),
                ("Médio Oriente", "médio oriente", 0.0045),
                ("Oriente continua", "oriente continua", 0.0117),
                ("registar-se violentos confrontos", "registar-se violentos confrontos", 0.0178),
                ("Faixa de Gaza", "faixa de gaza", 0.0268),
                ("fogo hoje voltaram", "fogo hoje voltaram", 0.0311),
                ("voltaram a registar-se", "voltaram a registar-se", 0.0311),
                ("registar-se violentos", "registar-se violentos", 0.0311),
                ("Exército israelita", "exército israelita", 0.0368),
                ("Exército israelita voltou", "exército israelita voltou", 0.0639),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn dataset_text_2_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_data_2.txt"),
            "en",
            Config::default(),
            5,
            [
                ("highly radioactive water", "highly radioactive water", 0.0006),
                ("crippled nuclear plant", "crippled nuclear plant", 0.0006),
                ("ocean Japan official", "ocean japan official", 0.0031),
                ("Japan official", "japan official", 0.0046),
                ("official says highly", "official says highly", 0.0050),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn dataset_text_3_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_data_3.txt"),
            "en",
            Config::default(),
            5,
            [
                ("Global Crossing", "global crossing", 0.0034),
                ("Hutchison Telecommunications", "hutchison telecommunications", 0.0053),
                ("Telecommunications and Singapore", "telecommunications and singapore", 0.0072),
                ("Singapore Technologies", "singapore technologies", 0.0072),
                ("Technologies take control", "technologies take control", 0.0157),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn dataset_text_4_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_data_4.txt"),
            "en",
            Config::default(),
            10,
            [
                ("annual revenues increasing", "annual revenues increasing", 0.0018),
                ("retail inventory management", "retail inventory management", 0.0023),
                ("Dollar General", "dollar general", 0.0034),
                ("inventory management", "inventory management", 0.0112),
                ("perpetual progress", "perpetual progress", 0.0133),
                ("revenues increasing", "revenues increasing", 0.0133),
                ("fast track", "fast track", 0.0133),
                ("road to perpetual", "road to perpetual", 0.0159),
                ("annual revenues", "annual revenues", 0.0168),
                ("stores opened", "stores opened", 0.0168),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn dataset_text_5_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_data_5.txt"),
            "en",
            Config::default(),
            10,
            [
                ("Handoff Trigger Table", "handoff trigger table", 0.0007),
                ("Handoff", "handoff", 0.0010),
                ("WLAN Networks ABSTRACT", "wlan networks abstract", 0.0019),
                ("Vertical handoff", "vertical handoff", 0.0020),
                ("Handoff Trigger", "handoff trigger", 0.0021),
                ("proactive handoff scheme", "proactive handoff scheme", 0.0021),
                ("HTT Method Figure", "htt method figure", 0.0022),
                ("WLAN", "wlan", 0.0023),
                ("ABSTRACT Vertical handoff", "abstract vertical handoff", 0.0030),
                ("traditional handoff scheme", "traditional handoff scheme", 0.0033),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn dataset_text_6_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_data_6.txt"),
            "en",
            Config::default(),
            10,
            [
                ("MRSA", "mrsa", 0.0047),
                ("TSN Database", "tsn database", 0.0107),
                ("methicillin-resistant Staphylococcus aureus", "methicillin-resistant staphylococcus aureus", 0.0116),
                ("rates of MRSA", "rates of mrsa", 0.0145),
                ("Staphylococcus aureus", "staphylococcus aureus", 0.0167),
                ("methicillin-resistant Staphylococcus", "methicillin-resistant staphylococcus", 0.0177),
                ("prevalence of MRSA", "prevalence of mrsa", 0.0201),
                ("MRSA infections", "mrsa infections", 0.0218),
                ("MRSA infections detected", "mrsa infections detected", 0.0223),
                ("TSN", "tsn", 0.0250),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }

    #[test]
    fn dataset_text_7_defaults() {
        // LIAAD/yake sample text
        test(
            include_str!("samples/test_data_7.txt"),
            "en",
            Config::default(),
            10,
            [
                ("Environment Design Level", "environment design level", 0.0008),
                ("Jerusalem Jerusalem", "jerusalem jerusalem", 0.0009),
                ("Dynamics Based Control", "dynamics based control", 0.0011),
                ("system dynamics", "system dynamics", 0.0017),
                ("DBC", "dbc", 0.0019),
                ("target system dynamics", "target system dynamics", 0.0019),
                ("target dynamics", "target dynamics", 0.0023),
                ("Science Bar Ilan", "science bar ilan", 0.0025),
                ("EMT", "emt", 0.0026),
                ("Dynamics", "dynamics", 0.0026),
            ],
        );
        // Results agree with reference implementation LIAAD/yake
    }
}
