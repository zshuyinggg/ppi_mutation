#%%
from tqdm import tqdm
import xml.etree.ElementTree as ET
from pandas.errors import ParserError
# get an iterable
# file = '/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/ClinVarFullRelease_00-latest.xml'
file = '/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/ClinVarFullRelease_00-latest.xml'
out = '/home/grads/z/zshuying/Documents/shuying/ppi_mutation/data/clinvar/organize_clinvar_3.txt'
# context = ET.iterparse(myfile, events=('start', 'end'))

context = ET.iterparse(file, events=('start', 'end'))

node = 0
node_measure = 0
pbar = tqdm(total=422796074)
clinvar_id, review_status, clinical_sig, uniprot_kb, variant_type, hgvs_p, missense = 0, 0, 0, 0, 0, 0, 0


with open(out, 'w') as f_out:
    f_out.write('clinvar_id,review_status,clinical_sig,clinvar_id,uniprot_kb,variant_type,hgvs_p,missense\n')
    try:
        for index, (event, elem) in enumerate(context):
            pbar.update(1)
            write_flag = 1
            # get the root element
            if index == 0:
                root = elem
            if event == 'start' and elem.tag == 'ClinicalSignificance':
                node = 1
            if event == 'end' and elem.tag == 'ClinicalSignificance':
                node = 0
            if event == 'start' and elem.tag == 'ClinVarAccession':
                clinvar_id = elem.attrib['Acc']
            # if event == 'end' and elem.tag == 'ReviewStatus' and node and ('reviewed by expert panel' in review_status or ):
            if event == 'end' and elem.tag == 'ReviewStatus' and node:
                review_status = elem.text 
                if 'no criteria' in elem.text or 'no assertion provided' in elem.text or 'no assertion criteria' in elem.text:review_status=0
            if event == 'end' and elem.tag == 'Description' and node:
                clinical_sig = elem.text
            if event == 'end' and elem.tag == 'XRef' and elem.attrib.get('DB') == 'UniProtKB':
                uniprot_kb = elem.attrib.get('ID')
            if event == 'start' and elem.tag == 'MeasureSet':
                node_measure = 1
            if event == 'end' and elem.tag == 'MeasureSet':
                node_measure = 0
            if node_measure and event == 'end':
                if elem.tag == 'Measure':
                    # print (elem.attrib['Type'])
                    # if variant_type!='single nucleotide variant':continue #some SNV are marked as "variation"
                    variant_type = elem.attrib.get('Type')
                    # elem.clear()
                if elem.tag == 'Attribute':
                    if elem.attrib.get('Type') == 'HGVS, protein':
                        # print(elem.text)
                        hgvs_p = elem.text
                    if elem.attrib.get('Type') == 'MolecularConsequence':
                        missense = elem.text
                        if missense != 'missense variant':
                            elem.clear()
                            # root.clear()
                            continue
            if event=='end' and elem.tag=='ClinVarSet':
                elem.clear()
                root.clear()
            if 0 in [clinvar_id, review_status, clinical_sig, variant_type, uniprot_kb,hgvs_p,
                     missense] or None in [clinvar_id, review_status, clinical_sig, variant_type, uniprot_kb,hgvs_p,
                     missense]: 
                # print(clinvar_id, review_status, clinical_sig, variant_type, uniprot_kb,
                #      missense)
                # elem.clear()
                # root.clear()     
                continue  # if the record misses the feature we need then skip it
            try:
                f_out.write(';'.join(
                    [clinvar_id, review_status, clinical_sig, uniprot_kb, variant_type, hgvs_p, missense]) + '\n')
                # print('\nwrote one line\n')
                clinvar_id, review_status, clinical_sig, uniprot_kb, variant_type, hgvs_p, missense = 0, 0, 0, 0, 0, 0, 0
                
            except TypeError:
                print([clinvar_id, review_status, clinical_sig, uniprot_kb, variant_type, hgvs_p, missense])
                clinvar_id, review_status, clinical_sig, uniprot_kb, variant_type, hgvs_p, missense = 0, 0, 0, 0, 0, 0, 0


            
        pbar.close()
    except SyntaxError:
        pass

# %%




# conflicting interpretations
# uncertain