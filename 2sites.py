from ED import *


M = 2  # how many sites?
Nu, Nd = 1,1  # make sure Nu,Nd <=M

SZ = (Nu-Nd)/2

bl = build_basis(M, Nu, Nd); Nb = len(bl)

shows(bl, M)  # show the basis

# load H00 ===============================
H00 = [[0,-1],[-1,0]]
H0 = load_H00(bl,H00,M)

# H0s = load_H00(bl,H00,M,sparse=True)
# print(np.any(H0-H0s))
# load HU ================================
U,JH = 7,0 # on-site U, Hund's coupling
sl = [[0],[1]]
Ul = [[U,JH],[U,JH]]
HU = load_H_Kanamori(bl, M, sl, Ul)


H = (H0+HU).toarray()
va,ve = np.linalg.eigh(H)

print('H=\n',H)

# find subspace labeled by S
# S2ve is the eigenvector of S^2 (with eigenvalues S(S+1))
# idx is the indexing information of block-diagonal matrix
S,idx,S2ve = analysis_S2(bl, M,prec=2,fraction=1)

# so we can perform a unitary transform S2ve.T.conj() x H x S2ve to get a 
# block-diagonal matrix, and extract each block by idx
Hbd = S2ve.T@H@S2ve
print('Hbd=\n',Hbd)


HS2 = [Hbd[i:j,i:j] for i,j in idx]
vaS = [np.linalg.eigvalsh(HS) for HS in  HS2]


for i in range(len(S)): print('S=%g:'%S[i], vaS[i][:5])


