from model_service import ModelService
from loguru import logger


@logger.catch
def main():
    gcn_multi = ModelService(multi_model=True)
    gcn_multi._set_globals()
    gcn_multi.load_models()
    smiles = ['CC(=O)N1CCCC1', 'CCCCCCCC(=O)OC']
    for smile in smiles:
        mean, sd = gcn_multi.get_preds(smile)
        print(f'{smile} SFE: {mean:.2f} ± {sd:.3f}')


if __name__ == '__main__':
    main()
