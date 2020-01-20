library(tidyverse)


# Zerilli et al 2016 ------------------------------------------------------


sheet1 =
  readxl::read_excel('~/Downloads/cll_data/leukemia/patient13.xlsx', sheet = 1) %>%
  mutate(VCF_CHR = str_replace(VCF_CHR, '^', 'chr')) %>%
  mutate(MUTID = str_c(VCF_CHR, VCF_BP1, VCF_A1...5, VCF_A1...6, sep = ':')) %>%
  mutate(r = round(FREQ * QDEPTH / 100)) %>%
  mutate(CNn = 2, CNt = 2, CNm = NA, PURITY = `purity estimates from FACS` / 100) %>%
  mutate(TP = case_when(TP == 'TP1' ~ 0,
                        TP == 'TP2' ~ 8,
                        TP == 'TP3' ~ 9,
                        TP == 'TP4' ~ 10,
                        TP == 'TP5' ~ 11,)) %>%
  mutate(SAMPLEID = str_c('S', as.integer(as.factor(TP)))) %>%
  select(SAMPLEID, TIME = TP, PURITY, MUTID, CHROM = VCF_CHR, POS = VCF_BP1,
         REF = VCF_A1...5, ALT = VCF_A1...6, GENE, TYPE = VAR_TYPE, r,
         R = QDEPTH, CNn, CNt, CNm) %>%
  view()

write_csv(sheet1, '~/Downloads/cll_data/cll_zerilli_2016_patient13.csv')

sheet2 = readxl::read_excel('~/Downloads/cll_data/leukemia/patient13.xlsx', sheet = 2)


# Rincon et al 2019 - Patient 1 ------------------------------------------------

sheet1 =
  readxl::read_excel('sci_rep_cll/patient1.xlsx', sheet = 1) %>%
  select(SAMPLEID = SAMPLE, TIME = `Sample date`, SAMPLE_TYPE = `Type of sample`,
         COMMENT = Commets, SEQ = Sequencing) %>%
  mutate(TIME = as.numeric(TIME - TIME[1], units = 'days')) %>%
  filter(!SAMPLEID %in% c('P1.14', 'P1.15'))

sheet2 =
  readxl::read_excel('sci_rep_cll/patient1.xlsx',
                     sheet = 3, na = '--') %>%
  drop_na() %>%
  select(CHROM = CHR, POS = POSITION, MUT = `DNA CHANGE`, CSQ = CONSEQUENCE, GENE,
         contains('total'), contains('variant')) %>%
  mutate(MUTID = str_c(CHROM, POS, MUT, sep = ':')) %>%
  gather(SAMPLE, VAL, contains('total'), contains('variant')) %>%
  extract(SAMPLE, into = c('SAMPLEID', 'VAR'), regex = '^(P1\\..+) (.+)$') %>%
  mutate(VAR = if_else(VAR == 'total', 'R', 'r')) %>%
  spread(VAR, VAL) %>%
  filter(!SAMPLEID %in% c('P1.14', 'P1.15'))

ds = left_join(sheet1, sheet2, by = 'SAMPLEID')

write_csv(ds, 'cll_Rincon_2019_patient1.csv')


# Rincon et al 2019 - Patient 2 ------------------------------------------------

sheet1 =
  readxl::read_excel('sci_rep_cll/patient2.xlsx', sheet = 1) %>%
  select(SAMPLEID = SAMPLE, TIME = `Sample date`, SAMPLE_TYPE = `Type of sample`,
         COMMENT = Commets) %>%
  mutate(TIME = as.numeric(TIME - TIME[1], units = 'days'))


sheet2 =
  readxl::read_excel('sci_rep_cll/patient2.xlsx',
                     sheet = 3, na = '--') %>%
  drop_na() %>%
  select(CHROM = CHR, POS = POSITION, MUT = `DNA CHANGE`, CSQ = CONSEQUENCE, GENE,
         contains('total'), contains('variant')) %>%
  mutate(MUTID = str_c(CHROM, POS, MUT, sep = ':')) %>%
  gather(SAMPLE, VAL, contains('total'), contains('variant')) %>%
  extract(SAMPLE, into = c('SAMPLEID', 'VAR'), regex = '^(P2\\..+) (.+)$') %>%
  mutate(VAR = if_else(VAR == 'total', 'R', 'r')) %>%
  spread(VAR, VAL) %>%
  mutate(r = as.numeric(if_else(r == 'nd', '0', r)),
         R = as.numeric(if_else(R == 'nd', '100', R)))

ds = left_join(sheet1, sheet2, by = 'SAMPLEID')

write_csv(ds, 'cll_Rincon_2019_patient2.csv')


# Cutts et al 2017 ------------------------------------------------------

depth =
  read_csv('melanoma/panel1_combined_ver_depth.csv') %>%
  filter(Plasma_006 > 10) %>%
  gather(SAMPLEID, R, Biopsy, starts_with('Plasma'))

vcfs =
  read_csv('melanoma/panel1_combined_ver.csv') %>%
  mutate_all(replace_na, 0.0) %>%
  gather(SAMPLEID, VCF, Biopsy, starts_with('Plasma'))

muts =
  left_join(depth, vcfs) %>%
  mutate(r = as.integer(round(VCF * R))) %>%
  mutate(MUTID = str_c(CHROM, POS, REF, ALT, sep = ':')) %>%
  # mutate(PURITY = 2 * mean(r / R, na.rm = T)) %>%
  mutate(PURITY = 0.1) %>%
  mutate(CNn = 2, CNt = 2) %>%
  mutate(TIME = case_when(SAMPLEID == 'Plasma_001' ~  -7,
                          SAMPLEID == 'Plasma_002' ~   0,
                          SAMPLEID == 'Plasma_003' ~  42,
                          SAMPLEID == 'Plasma_004' ~  63,
                          SAMPLEID == 'Plasma_005' ~ 119,
                          SAMPLEID == 'Plasma_006' ~ 161,
                          SAMPLEID == 'Plasma_007' ~ 280,
                          SAMPLEID == 'Plasma_008' ~ 343,
                          SAMPLEID == 'Plasma_009' ~ 378,
                          SAMPLEID == 'Plasma_010' ~ 384)) %>%
  filter(SAMPLEID != 'Biopsy') %>%
  select(SAMPLEID, TIME, PURITY, MUTID, CHROM, POS, REF, ALT, GENE, R, r, CNn, CNt)

write_csv(muts, '../melanoma_Cutts_2017.csv')



# Schuh et al, 2012
#write_lines(pdftools::pdf_text('Desktop/CLL077.pdf'), 'Desktop/CLL077.txt')

