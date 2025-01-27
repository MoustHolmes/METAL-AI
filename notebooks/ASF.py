from dataclasses import dataclass
import numpy as np
import hashlib
import periodictable

@dataclass
class Hyperparams:
    # type: list of tuples of (int,int)
    orbitals: list[tuple[int,int]]           # orbitals as (n,l), ordered by filling order
    num_excitations: int     # number of electrons to produce excitation level of
    max_excitation: int      # maximum level to excite to

    @classmethod
    def get_default(self):
        l_map = {"s":0, "p":1, "d":2, "f":3, "g":4}
        orbitals = ["1s", 
                    "2s", 
                    "2p", "3s", 
                    "3p", "4s", 
                    "3d", "4p", "5s", 
                    "4d", "5p", "6s", 
                    "4f", "5d", "6p", "7s", 
                    "5f", "6d", "7p", "8s",
                    "5g", "6f", "7d", "8p", "9s"]
        orbitals = [(int(n), l_map[l]) for n,l in orbitals]
        return Hyperparams(orbitals=orbitals, num_excitations=4, max_excitation=8)

# class for converting between excitation representation and filling number representation
class ASF:
    def __init__(self, num_electrons, num_protons, excitations, hyperparams = Hyperparams.get_default()) -> None:
        self.num_electrons = num_electrons      # number of electrons
        self.num_protons = num_protons          # number of protons
        self.hyperparams = hyperparams
        self.__make_default_CSF(num_electrons)
        self.excitations = frozenset(tuple(x) for x in excitations)

        valid = [len(x) == hyperparams.num_excitations and max(x) < hyperparams.max_excitation for x in excitations]
        if not all(valid):
            invalid_index = valid.index(False)
            raise ValueError(f"Excitation {invalid_index} is invalid: {list(excitations)[invalid_index]}")

        self.filling_numbers = frozenset(tuple(self.to_filling_number(x)) for x in excitations)


    def __make_default_CSF(self, num_electrons):
        self.default_CSF = []
        self.default_holes = []
        self.default_mobile_levels = []
        for orb in self.hyperparams.orbitals:
            level_size = 2*(1+2*orb[1])
            added_electrons = min(level_size, num_electrons)
            self.default_CSF.append(added_electrons)
            self.default_holes.append(level_size - added_electrons)
            num_electrons -= added_electrons
            mobile_electrons = min(self.hyperparams.num_excitations - num_electrons, added_electrons)
            if mobile_electrons:
                self.default_mobile_levels.extend([len(self.default_CSF)-1] * mobile_electrons)

    def add_excitation(self, excitation):
        return ASF(self.num_electrons, self.num_protons, self.excitations | {excitation}, self.hyperparams)

    def to_filling_number(self, excitation):
        CSF = self.default_CSF.copy()
        holes = self.default_holes.copy()
        # for each electron listed here, move them the given amount of levels up
        # question: Should invalid states be rejected, or made valid?
        for lvl, excitation in zip(self.default_mobile_levels, excitation):
            holes[lvl] += 1
            CSF[lvl] -= 1
            lvl += excitation
            if lvl >= len(CSF):
                print(f"Warning: Excitation {excitation} is invalid for {self.num_electrons} electrons and {self.num_protons} protons")
            while (holes[lvl] == 0):
                lvl+=1
            holes[lvl] -= 1
            CSF[lvl] += 1
        return CSF
    

    def to_string(self):
        L_names = "SPDFG"
        orbitals = 0
        output = ""
        filling_numbers_ordered = np.atleast_2d(sorted(self.filling_numbers))
        for (n,l), level in zip(self.hyperparams.orbitals, filling_numbers_ordered.T):
            if np.all(level == 0):
                continue
            if np.any(level != level[0]):
                fillings = ' '.join(map(str, level))
            else:
                fillings = level[0]
            orbitals += 1
            output += f" {n}{L_names[l]}  {fillings}\n"
        header = f" {len(self.filling_numbers)} {orbitals} {0} ! {self.num_electrons} {self.num_protons}\n"
        return header + output
    
    # gets all excitations that are unique, i.e. that lead to different CSFs
    @staticmethod
    def get_unique_excitations(num_electrons, num_protons, hyperparams, excitations_unfiltered):
        # get a default CSF to which we can add excitations
        asf = ASF(num_electrons=num_electrons, num_protons=num_protons, excitations=[], hyperparams=hyperparams)
        excitations = []
        # loop over all excitations and add them to the default CSF, tracking which ones are unique
        for excitation in excitations_unfiltered:
            asf_new = asf.add_excitation(excitation)
            if not asf == asf_new:
                excitations.append(excitation)
                asf = asf_new
        return excitations
    
    def get_uuid(csf) -> str:
        return hashlib.sha512(csf.to_string().encode()).hexdigest()
        
    def get_name(self):
        charge = self.num_protons - self.num_electrons
        name = periodictable.elements[self.num_protons].symbol
        if charge < 4:
            name += "I" * charge
        if charge == 4: 
            name += "IV" 
        if charge == 4: 
            name += "V" 
        return name
    
    # return a hash of the set of CSFs, which is the same for two CSFs if their
    # excitations are the same, irrespective of the order of the excitations
    def __hash__(self) -> int:
        return hash((self.num_electrons, self.num_protons, self.filling_numbers))
    
    def __eq__(self, o: object) -> bool:
        return self.filling_numbers == o.filling_numbers and self.num_electrons == o.num_electrons and self.num_protons == o.num_protons