import { Product } from "@/_models/Product";
import { atom } from "jotai";

export const currentProductAtom = atom<Product | undefined>(undefined);

export const selectedModelAtom = atom<Product | undefined>(undefined);